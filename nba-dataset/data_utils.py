from __future__ import absolute_import, division, print_function

import os
import pandas as pd
import numpy as np
from glob import glob
import cPickle as pickle

import sys
sys.path.append("../code")
from stg_node import *
from experiment_details import NUM_DATAFILES, ROWS_TO_EXTRACT

from Event import Event


def split_files(all_files, train_eval_test_split=[.85, .15, 0.], seed=1234):
    np.random.seed(seed)
    np.random.shuffle(all_files)

    train_cutoff, eval_cutoff, test_cutoff = map(int, np.cumsum(train_eval_test_split)*len(all_files))

    train_files = all_files[:train_cutoff]
    eval_files = all_files[train_cutoff:eval_cutoff]
    if eval_cutoff == test_cutoff:
        test_files = None
    else:
        test_files = all_files[eval_cutoff:]

    return train_files, eval_files, test_files


def get_pos_dict(files, positions_map_path=None, rows_to_extract=ROWS_TO_EXTRACT):    
    positions_map = load_positions_map(positions_map_path)

    pos_dict = dict()
    for idx, f_name in enumerate(files):
        game_name = os.path.basename(os.path.splitext(f_name)[0])
        print('[get_pos_dict] Reading ' + game_name, end='... ')
        
        data_frame = pd.read_json(f_name)
        
        if rows_to_extract == 'all':
            num_rows = len(data_frame)
        else:
            num_rows = rows_to_extract
        
        for index in xrange(num_rows):
            event = Event(data_frame['events'][index], positions_map=positions_map)
            
            curr_feat_dict = event.get_features_dict(only_initial=True)
            if len(curr_feat_dict) == 0:
                continue
            
            pos_dict[game_name + '/' + event.id] = {(node.name, node.type): (curr_feat_dict[node][0, 0], curr_feat_dict[node][0, 1]) for node in curr_feat_dict}
            
        print('Done! %d/%d' % (idx + 1, len(files)))
    
    return pos_dict


def load_positions_map(positions_map_path):
    if positions_map_path is None:
        return None
    
    with open(positions_map_path, 'rb') as f:
        positions_map = pickle.load(f)
        
    return positions_map


def get_basketball_dicts(files, positions_map_path=None, rows_to_extract=ROWS_TO_EXTRACT):
    positions_map = load_positions_map(positions_map_path)
    
    feature_dicts = dict()
    max_times = list()
    num_states = list()
    all_players = set()
    for idx, f_name in enumerate(files):
        game_name = os.path.basename(os.path.splitext(f_name)[0])
        print('[get_basketball_dicts] Reading ' + game_name, end='... ')
        
        data_frame = pd.read_json(f_name)
        
        if rows_to_extract == 'all':
            num_rows = len(data_frame)
        else:
            num_rows = rows_to_extract

        for index in xrange(num_rows):
            event = Event(data_frame['events'][index], positions_map=positions_map)
            
            curr_feat_dict, avg_time_diff = event.get_features_dict(return_avg_time_diff=True)
            if len(curr_feat_dict) == 0:
                continue
            
            (T, S) = curr_feat_dict.values()[0].shape
            max_times.append(T)
            num_states.append(S)
            all_players.update(curr_feat_dict.keys())
            
            # game + eventId = bag
            feature_dicts[game_name + '/' + event.id] = (curr_feat_dict, T, avg_time_diff)
            
        print('Done! %d/%d' % (idx + 1, len(files)))
    
    nbags = len(feature_dicts)
    tmax = max(max_times) # tmax = max_actual_time + 1
    nstates = max(num_states) # This should almost always be 6
    all_players = list(all_players)
    
    player_arrays = dict()
    for player in all_players:
        player_arrays[player] = np.zeros([nbags, tmax, nstates])
    
    print('[get_basketball_dicts] Forming useful features dict', end='... ')
    trajectory_lengths = np.zeros(nbags, dtype=np.int32)
    bag_idx = -np.ones([nbags, tmax, 1], dtype=np.int32)
    bag_names = feature_dicts.keys()
    avg_time_diffs = np.zeros(nbags, dtype=np.float32)
    for i, bag in enumerate(feature_dicts.keys()):
        (bag, bag_length, avg_time_diff) = feature_dicts[bag]
        for player_name in bag.keys():
            player_data = bag[player_name]
            player_arrays[player_name][i, :bag_length, :] = player_data
            
        trajectory_lengths[i] = bag_length
        bag_idx[i, :bag_length, 0] = i
        avg_time_diffs[i] = avg_time_diff
    
    useful_features_dict = player_arrays
    useful_features_dict["traj_lengths"] = trajectory_lengths
    useful_features_dict["bag_idx"] = bag_idx
    useful_features_dict["bag_names"] = bag_names
    useful_features_dict["avg_time_diffs"] = avg_time_diffs
    # This is _meant_ to be zero-length on the last dimension to 
    # match our absense of extra information!
    useful_features_dict["extras"] = np.zeros([nbags, tmax, 0])
    print('Done!')
    
    return useful_features_dict


def get_data_dict(files, positions_map_path=None, pred_indices=[2, 3], rows_to_extract=ROWS_TO_EXTRACT):
    """pred_indices: The indices we want to predict. [2, 3] are the 
                     indices for the x and y velocity, respectively. 
    """
    
    def extract_mean_and_std(A, tl, off=0):
        data = np.concatenate([A[i,:l+off,:] for i, l in enumerate(tl)], axis=0)
        return data.mean(axis=0).astype(np.float32), data.std(axis=0).astype(np.float32)

    useful_features_dict = get_basketball_dicts(files, 
                                                positions_map_path=positions_map_path, 
                                                rows_to_extract=rows_to_extract)
    all_nodes = [key for key in useful_features_dict if isinstance(key, STGNode)]

    bag_names = useful_features_dict.pop("bag_names")
    
    data_dict = {}
    data_dict["input_dict"] = useful_features_dict
    data_dict["bag_names"] = bag_names
    # What do we with this? Is this ok for multiple outputs?
    data_dict["labels"] = {convert_to_label_node(node): data_dict["input_dict"][node][:,:,pred_indices] for node in all_nodes}
    data_dict["pred_indices"] = pred_indices
    
    traj_lengths = data_dict["input_dict"]["traj_lengths"]
    data_dict["extras_mean"], data_dict["extras_std"] = extract_mean_and_std(data_dict["input_dict"]["extras"],
                                                                             traj_lengths)
    nodes_standardization_dict = dict()
    labels_standardization_dict = dict()
    for node in all_nodes:
        mean, std = extract_mean_and_std(data_dict["input_dict"][node], traj_lengths)
        nodes_standardization_dict[node] = {"mean": mean, "std": std}
        
        label_node = convert_to_label_node(node)
        mean, std = extract_mean_and_std(data_dict["labels"][label_node], traj_lengths)
        labels_standardization_dict[label_node] = {"mean": mean, "std": std}
            
    data_dict["nodes_standardization"] = nodes_standardization_dict
    data_dict["labels_standardization"] = labels_standardization_dict
    
    return data_dict
