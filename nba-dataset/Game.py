import pandas as pd
import numpy as np

from Event import Event
from Team import Team
from Constant import Constant


class Game:
    """A class for keeping info about the games"""
    def __init__(self, path_to_json, event_index=0):
        # self.events = None
        self.home_team = None
        self.guest_team = None
        self.event = None
        self.path_to_json = path_to_json
        self.event_index = event_index

    def read_json(self):
        data_frame = pd.read_json(self.path_to_json)
        last_default_index = len(data_frame) - 1
        self.event_index = min(self.event_index, last_default_index)
        index = self.event_index

        print(Constant.MESSAGE + str(last_default_index))
        event = data_frame['events'][index]
        self.event = Event(event)
        self.home_team = Team(event['home']['teamid'])
        self.guest_team = Team(event['visitor']['teamid'])

    def start(self):
        self.event.show()

    def get_feature_dict(self):
        return self.event.get_features_dict()

    def _convert_feature_dict_pos_matrix(self, feature_dict, node_names=None):
        if node_names is None:
            node_names = feature_dict.keys()

        N, T = len(node_names), feature_dict.values()[0].shape[0]
        pos_matrix = np.zeros((T, N, 2))
        for i, node_name in enumerate(node_names):
            pos_matrix[:, i, :] = feature_dict[node_name][:, :2]

        return pos_matrix
    
    def get_pos_matrix(self, return_feature_dict=False):
        feature_dict = self.get_feature_dict()
        if return_feature_dict:
            return self._convert_feature_dict_pos_matrix(feature_dict), feature_dict
        else:
            return self._convert_feature_dict_pos_matrix(feature_dict)
    
    def get_st_graph_info(self, robot_node_name):
        pos_matrix, feature_dict = self.get_pos_matrix(return_feature_dict=True)
        node_names = feature_dict.keys()
        type_list = [self.event.player_types[node_name] for node_name in node_names]
        robot_node_type = self.event.player_types[robot_node_name]
        
        return pos_matrix[0], type_list, node_names, robot_node_name, robot_node_type            
