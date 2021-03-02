import os

from dgl import save_graphs, load_graphs
from dgl.data import DGLDataset
from dgl.data.utils import save_info, load_info
from torch.utils.data import Dataset

from src.utils import PROJECT_PATH

DGL_PATH = os.path.join(PROJECT_PATH, "data", "processed", "DGL")
if not os.path.isdir(DGL_PATH):
    os.mkdir(DGL_PATH)

class BaseDataset(DGLDataset):
    """ DGL dataset for Whole Body Human Motion Dataset
    Check if the data with a specific configuration is existed on disk
    If we have the data => Load the processed data
    If not => process data and save cache
    
    Args:

    """
      
    def __init__(self, name, data_dir, force_reload=False, verbose=False, 
                 obs_len=10, pred_len=20, is_test=False, **kwargs):
        
        graph_name = "H36M_{}_obs{}_pred{}".format(name, obs_len, pred_len)
        self.graph_name = graph_name
        
        self.is_test = is_test
        if is_test:
            self.subjects = ["S5"]
        else:
            self.subjects = ['S1','S6','S7','S8','S9','S11']
        
        self.config = {
            "obs_len": obs_len,
            "pred_len": pred_len,
        }
        
        self.additional_info = {
            "graph_infos": [],
            "frame_idx": [],
        }
        
        self.data_dir = data_dir
                
        super().__init__(name=graph_name, url=None, raw_dir=None, force_reload=force_reload, verbose=verbose)
    
    def download(self):
        pass
    
    def process(self):
        pass
    
    def __getitem__(self, idx):
        return self.graphs[idx], idx
        
    def __len__(self):
        num_items = len(self.graphs)
        num_items = int(num_items * (1 - self.config["drop"]))
        
        return num_items
    
    def save(self):
        graph_path = os.path.join(DGL_PATH, self.graph_name + "_dgl_graph.bin")
        save_graphs(graph_path, self.graphs)
        
        info_path = os.path.join(DGL_PATH, self.graph_name + "_info.pkl")

        save_info(info_path, {
            "config": self.config,    
            "additional_info": self.additional_info      
        })
        
    def load(self):
        graph_path = os.path.join(DGL_PATH, self.graph_name + "_dgl_graph.bin")
        self.graphs, _ = load_graphs(graph_path)
        
        info_path = os.path.join(DGL_PATH, self.graph_name + "_info.pkl")
        info = load_info(info_path)
        
        self.config = info["config"]
        self.additional_info = info["additional_info"]
 
    # def get_subset(self, indices, augment=False):
    #     return SubsetDataset(self.config, self.additional_info,
    #                          self.graphs, indices)
                
    def has_cache(self):
        graph_path = os.path.join(DGL_PATH, self.graph_name + "_dgl_graph.bin")
        info_path = os.path.join(DGL_PATH, self.graph_name + "_info.pkl")

        info_same = os.path.exists(info_path)
        if info_same:
            saved_info = load_info(info_path)
            
            info_same = info_same and (saved_info["config"] == self.config)
            
        return os.path.exists(graph_path) and info_same