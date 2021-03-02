import os
import copy

import dgl
from dgl.data import load_data
import torch
import numpy as np

from src.features.h36m import BaseDataset
import src.features.h36m_utils as data_utils

actions = ["walking", "eating", "smoking"]
subactions = ["1", "2"]
rng = np.random.RandomState(1234567890)

nodeFeaturesRanges={}
nodeFeaturesRanges['torso'] = range(6)
nodeFeaturesRanges['torso'].extend(range(36,51))
nodeFeaturesRanges['right_arm'] = range(75,99)
nodeFeaturesRanges['left_arm'] = range(51,75)
nodeFeaturesRanges['right_leg'] = range(6,21)
nodeFeaturesRanges['left_leg'] = range(21,36)


class H36MDataset(BaseDataset):
    def __init__(self, name, **kwargs):
        super().__init__(name="srnn_{}".format(name), **kwargs)
        
    def process(self):        
        raw_data, completeData = data_utils.load_data(self.data_dir, subjects=self.subjects, 
                                                       actions=actions, one_hot=False)
        
        
        if not self.is_test:
            data_mean, data_std, dim_to_ignore, dim_to_use = data_utils.normalization_stats(completeData)
            self.data_stat = {
                "data_mean": data_mean,
                "data_std": data_std,
                "dim_to_ignore": dim_to_ignore,
                "dim_to_use": dim_to_use
            }
        else:
            data_mean = self.data_stat["data_mean"]
            data_std = self.data_stat["data_std"]
            dim_to_ignore = self.data_stat["dim_to_ignore"]
            dim_to_use = self.data_stat["dim_to_use"]
        
        raw_data = data_utils.normalize_data( raw_data, data_mean, data_std, dim_to_use, actions, one_hot=False)
        obs_len, pred_len = self.config["obs_len"], self.config["pred_len"]
        
        