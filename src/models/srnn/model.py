import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import dgl   

"""
Algo:
- Initialize h, set all to zeros
- For each spatial edge type in the graph:
+ Compute distance
+ Put the distance to GRU
+ Update the hidden state of the invovled nodes
- Update temporal edge RNN
- Concatenate all hidden state
- Update node RNN

"""

def d_object_human(src, dst):
    b_size, seq_len, h_dim = dst.size()
    _, _, o_dim = src.size()
    
    # batch_size, seq_len, 18, 3
    human = dst.reshape(b_size, seq_len, -1, 3)
    
    # batch_size, seq_len, 8, 3
    obj = src.reshape(b_size, seq_len, -1, 3)
    
    num_h_points = human.size(2)
    num_o_points = obj.size(2)
    
    human = human.repeat(1, 1, num_o_points, 1)
    obj = obj.unsqueeze(3).repeat(1, 1, 1, num_h_points, 1).reshape(b_size, seq_len, -1, 3)

    return (human - obj).reshape(b_size, seq_len, -1)

def d_object_object(src, dst):
    return dst - src

def d_human_human(src, dst):
    return dst - src
    
class Model(nn.Module):
    def __init__(self, human_dim, object_dim, 
                 human_node_rnn_dim, human_tedge_rnn_dim,
                 human_sedge_object_rnn_dim, human_sedge_human_rnn_dim,
                 object_node_rnn_dim, object_tedge_rnn_dim,
                 object_sedge_object_rnn_dim):
        super().__init__()
        
        self.human_dim = human_dim
        self.node_h_dim = {
            "human": human_node_rnn_dim,
            "object": object_node_rnn_dim
        }
        
        self.sedge_h_dim = {
            "human, object": human_sedge_object_rnn_dim,
            "human, human": human_sedge_human_rnn_dim,
            "object, object": object_sedge_object_rnn_dim
        }
        
        self.tedge_h_dim = {
            "human": human_tedge_rnn_dim,
            "object": object_tedge_rnn_dim
        }
                
        self.spatial_edge_rnn = nn.ModuleDict({
            "human, object": nn.GRU(8 * human_dim, human_sedge_object_rnn_dim),
            "human, human": nn.GRU(human_dim, human_sedge_human_rnn_dim),
            "object, object": nn.GRU(object_dim, object_sedge_object_rnn_dim)
        })
        
        self.temporal_edge_rnn = nn.ModuleDict({
            "human": nn.GRU(human_dim, human_tedge_rnn_dim),
            "object": nn.GRU(object_dim, object_tedge_rnn_dim)
        })
        
        self.node_rnn = nn.ModuleDict({
            "human": nn.GRU(human_sedge_human_rnn_dim + human_sedge_object_rnn_dim + human_tedge_rnn_dim + human_dim,
                            human_node_rnn_dim),
            "object": nn.GRU(object_sedge_object_rnn_dim + human_sedge_object_rnn_dim + object_tedge_rnn_dim + object_dim,
                             object_node_rnn_dim)
        })
        
        self.out = nn.ModuleDict({
            "human": nn.Linear(human_node_rnn_dim, human_dim),
            "object": nn.Linear(object_node_rnn_dim, object_dim)
        })
        
    def initialize_hidden(self, batch_size, dim, device):
        return torch.zeros(batch_size, dim).to(device)
                
    def initialize_spatial_edge_rnn(self, graph):
        for etype in sorted(graph.canonical_etypes):
            srctype, dsttype = etype[0], etype[2]
            
            if srctype == "object" and dsttype == "human":
                name = "human, object"
            else:
                name = "{}, {}".format(srctype, dsttype)
            
            num_nodes = graph.num_nodes(dsttype)
            graph.nodes[dsttype].data[name] = self.initialize_hidden(num_nodes, 
                                                                     self.sedge_h_dim[name],
                                                                     graph.device)
        
        return graph
    
    def initialize_temporal_edge_rnn(self, graph):
        graph.nodes["human"].data["temporal"] = self.initialize_hidden(graph.num_nodes("human"),
                                                                       self.tedge_h_dim["human"],
                                                                       graph.device)
        graph.nodes["object"].data["temporal"] = self.initialize_hidden(graph.num_nodes("object"),
                                                                        self.tedge_h_dim["object"],
                                                                        graph.device)

    def initialize_node_rnn(self, graph):
        graph.nodes["human"].data["node"] = self.initialize_hidden(graph.num_nodes("human"),
                                                                   self.node_h_dim["human"],
                                                                   graph.device)
        
        graph.nodes["object"].data["node"] = self.initialize_hidden(graph.num_nodes("object"),
                                                                    self.node_h_dim["object"],
                                                                    graph.device)

    def oh_msg_gen(self, phase="obs"):
        def oh_msg(edges):
            if phase=="obs":
                d = d_object_human(edges.src["x"], edges.dst["x"])
            elif phase=="pred":
                d = d_object_human(edges.src["y_hat"], edges.dst["y_hat"])
        
            return {"d": d}
        
        return oh_msg
    
    def ho_msg_gen(self, phase="obs"):
        def ho_msg(edges):
            if phase=="obs":
                d = -1 * d_object_human(edges.dst["x"], edges.src["x"])
            elif phase=="pred":
                d = -1 * d_object_human(edges.dst["y_hat"], edges.src["y_hat"])
        
            return {"d": d}
        
        return ho_msg
    
    def oo_msg_gen(self, phase="obs"):
        def oo_msg(edges):
            if phase == "obs":
                d = edges.dst["x"] - edges.src["x"]
            elif phase == "pred":
                d = edges.dst["y_hat"] - edges.src["y_hat"]
                
            return {"d": d}
        
        return oo_msg
    
    def hh_msg_gen(self, phase="obs"):
        def hh_msg(edges):
            if phase == "obs":
                d = edges.dst["x"] - edges.src["x"]
            elif phase == "pred":
                d = edges.dst["y_hat"] - edges.src["y_hat"]
            return {"d": d}
        
        return hh_msg

    def reduce_gen(self, out_name):
        def reduce_func(nodes):
            m = torch.sum(nodes.mailbox["d"], 1)
            return {out_name: m}
        
        return reduce_func

    def observe(self, graph):
        obs_len = graph.nodes["human"].data["x"].size(1)
        node_rnn_inputs = {
            "object": {
            },
            "human": {
            }
        }
        
        for etype in sorted(graph.canonical_etypes):
            srctype, dsttype = etype[0], etype[2]
            
            if srctype == "object" and dsttype == "human":
                current_edge = "human, object"
                msg_fn = self.oh_msg_gen("obs")
            
            elif srctype == "human" and dsttype == "human":
                current_edge = "human, human"
                msg_fn = self.hh_msg_gen("obs")

            elif srctype == "human" and dsttype == "object":
                current_edge = "human, object"
                msg_fn = self.ho_msg_gen("obs")

            elif srctype == "object" and dsttype == "object":
                current_edge = "object, object"
                msg_fn = self.oo_msg_gen("obs")

            graph.update_all(msg_fn,
                             self.reduce_gen("m_{}".format(current_edge)),
                             etype=etype)
            
            edata = graph.nodes[dsttype].data.pop("m_{}".format(current_edge))
            edata = edata.permute(1, 0, 2)
            
            h = graph.nodes[dsttype].data[current_edge]
            o, h = self.spatial_edge_rnn[current_edge](edata, h.unsqueeze(0))
            graph.nodes[dsttype].data[current_edge] = h[-1]
            
            node_rnn_inputs[dsttype][current_edge] = o
        
        for ntype in sorted(graph.ntypes):
            current_node = graph.nodes[ntype]
            x = current_node.data["x"]
            
            vel = torch.zeros_like(x)
            vel[:, 1:] = vel[:, 1:] - vel[:, :-1]
            vel = vel.permute(1, 0, 2)
            
            temp_h = current_node.data["temporal"]
            temp_o, temp_h = self.temporal_edge_rnn[ntype](vel, temp_h.unsqueeze(0))
            graph.nodes[ntype].data["temporal"] = temp_h[-1]
            
            if ntype == "human":
                inp = [x.permute(1, 0, 2), temp_o, 
                       node_rnn_inputs[ntype]["human, object"], 
                       node_rnn_inputs[ntype]["human, human"]]
                
            elif ntype == "object":
                inp = [x.permute(1, 0, 2), temp_o, 
                       node_rnn_inputs[ntype]["human, object"], 
                       node_rnn_inputs[ntype]["object, object"]]

            inp = torch.cat(inp, dim=2)
            h = current_node.data["node"]
            
            o, h = self.node_rnn[ntype](inp, h.unsqueeze(0))
            
            graph.nodes[ntype].data["node"] = h[-1]

    def predict(self, graph, pred_len):
        for ntype in sorted(graph.ntypes):
            graph.nodes[ntype].data["y_hat"] = graph.nodes[ntype].data["x"][:, -1].clone().unsqueeze(1)
        out = {
            "human": [],
            "object": [],
        }
        
        velocity = {
            "human": None,
            "object": None
        }
        
        for t in range(pred_len):
            for ntype in sorted(graph.ntypes):
                current_node = graph.nodes[ntype]
                vel = self.out[ntype](current_node.data["node"]).unsqueeze(1)
                velocity[ntype] = vel

                y_hat = current_node.data["y_hat"] + vel
                
                out[ntype].append(y_hat.clone())
                graph.nodes[ntype].data["y_hat"] = y_hat

            for etype in sorted(graph.canonical_etypes):
                srctype, dsttype = etype[0], etype[2]
                
                if srctype == "object" and dsttype == "human":
                    current_edge = "human, object"
                    msg_gen = self.oh_msg_gen("pred")
                
                elif srctype == "human" and dsttype == "human":
                    current_edge = "human, human"
                    msg_gen = self.hh_msg_gen("pred")

                elif srctype == "human" and dsttype == "object":
                    current_edge = "human, object"
                    msg_gen = self.ho_msg_gen("pred")

                elif srctype == "object" and dsttype == "object":
                    current_edge = "object, object"
                    msg_gen = self.oo_msg_gen("pred")

                graph.update_all(msg_gen,
                                self.reduce_gen("m_{}".format(current_edge)),
                                etype=etype)
                
                edata = graph.nodes[dsttype].data.pop("m_{}".format(current_edge))
                edata = edata.permute(1, 0, 2)
                
                h = graph.nodes[dsttype].data[current_edge]
                o, h = self.spatial_edge_rnn[current_edge](edata, h.unsqueeze(0))
                graph.nodes[dsttype].data[current_edge] = h[-1]

            for ntype in sorted(graph.ntypes):
                current_node = graph.nodes[ntype]
                x = current_node.data["y_hat"][:, -1]
                vel = velocity[ntype].permute(1, 0, 2)
                
                temp_h = current_node.data["temporal"]
                _, temp_h = self.temporal_edge_rnn[ntype](vel, temp_h.unsqueeze(0))
                graph.nodes[ntype].data["temporal"] = temp_h[-1]
                
                if ntype == "human":
                    inp = [x, temp_h[-1], 
                           current_node.data["human, object"], 
                           current_node.data["human, human"]]
                    
                elif ntype == "object":
                    inp = [x, temp_h[-1], 
                           current_node.data["human, object"], 
                           current_node.data["object, object"]]

                inp = torch.cat(inp, dim=1).unsqueeze(0)
                h = current_node.data["node"]
                
                o, h = self.node_rnn[ntype](inp, h.unsqueeze(0))
                
                graph.nodes[ntype].data["node"] = h[-1]

        for ntype in out:
            out[ntype] = torch.cat(out[ntype], dim=1)

        return out["human"], out["object"]

    def forward(self, graph, pred_len=20):
        self.initialize_spatial_edge_rnn(graph)
        self.initialize_temporal_edge_rnn(graph)
        self.initialize_node_rnn(graph)
        
        self.observe(graph)
        return self.predict(graph, pred_len)