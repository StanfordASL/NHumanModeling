from __future__ import absolute_import, division, print_function
import logging
import sys
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', 
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO, 
                    stream=sys.stderr)

import json
import os
if os.path.isfile("../../code/config.json"):
    with open("../../code/config.json", "r") as f:
        config = json.load(f)
        config['data_dir'] = '../' + config['data_dir']
        config['julia_pkg_dir'] = '../' + config['julia_pkg_dir']
        config['models_dir'] = '../' + config['models_dir']
else:
    logging.error("Please run setup.py in this directory before running any .ipynb's.")

_ARGS_LENGTH = 7
if len(sys.argv) == _ARGS_LENGTH:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from glob import glob
import tensorflow as tf
import time
import shutil
import cPickle as pickle
from collections import OrderedDict, defaultdict

from tensorflow.python.client import device_lib
logging.info(device_lib.list_local_devices())

sys.path.append("../../code")
sys.path.append("..")
from utils.bags import *
from utils.learning import *
from traffic_weaving_models import *
from st_graph import *
from data_utils import *
from stg_node import *

from experiment_details import get_output_ckpts_dir_name
if len(sys.argv) == _ARGS_LENGTH:
    NUM_DATAFILES = int(sys.argv[2])
    ROWS_TO_EXTRACT = sys.argv[3]
    EDGE_RADIUS = float(sys.argv[4])
    EDGE_STATE_COMBINE_METHOD = sys.argv[5]
    EDGE_INFLUENCE_COMBINE_METHOD = sys.argv[6]
else:
    from experiment_details import NUM_DATAFILES, ROWS_TO_EXTRACT, EDGE_RADIUS, EDGE_STATE_COMBINE_METHOD, EDGE_INFLUENCE_COMBINE_METHOD

logging.info('NUM_DATAFILES = %d' % NUM_DATAFILES)
logging.info('ROWS_TO_EXTRACT = %s' % str(ROWS_TO_EXTRACT))


def memory(prefix=''):
    import psutil
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
    logging.info(prefix + ' memory use: ' + str(memoryUse))
    

class Runner(object):
    
    def setup(self):
        logging.info(config)

        model_dir = os.path.join(config["models_dir"], get_output_ckpts_dir_name(NUM_DATAFILES, 
                                                                                 ROWS_TO_EXTRACT, 
                                                                                 EDGE_RADIUS,
                                                                                 EDGE_STATE_COMBINE_METHOD,
                                                                                 EDGE_INFLUENCE_COMBINE_METHOD))
        
        #logging.warn('Deleting %s!' % model_dir)
        #shutil.rmtree(model_dir, ignore_errors=True)

        sc = tf.ConfigProto(device_count={'GPU': 1}, 
                            allow_soft_placement=True)
        rc = tf.estimator.RunConfig().replace(session_config=sc, 
                                              model_dir=model_dir,
                                              save_summary_steps=10,
                                              save_checkpoints_steps=10,
                                              log_step_count_steps=10,
                                              keep_checkpoint_max=None,
                                              tf_random_seed=None)
        
        # required due to a bug in tf.contrib.learn.Experiment.train_and_evaluate
        rc.environment = None
        
        self.rc = rc
        self.model_dir = model_dir
        
        logging.info(model_dir)
        logging.info('[setup] Done!')


    def load_data_and_define_model(self):
        data_dir = os.path.join(config['data_dir'], "2016.NBA.Raw.SportVU.Game.Logs")
        all_files = glob(os.path.join(data_dir, '*.json'))[:NUM_DATAFILES]
        train_files, eval_files, test_files = split_files(all_files, train_eval_test_split=[.8, .2, .0], seed=123)

        self.robot_node = robot_node = STGNode('Al Horford', 'HomeC')
        positions_map_path = os.path.join(config['data_dir'], "positions_map.pkl")
        pos_dict_path = os.path.join(config['data_dir'], "pos_dict_%d_files_%s_rows.pkl" % (NUM_DATAFILES, str(ROWS_TO_EXTRACT)))

        if os.path.isfile(pos_dict_path):
            with open(pos_dict_path, 'rb') as f:
                pos_dict = pickle.load(f)
        else:
            pos_dict = get_pos_dict(train_files, positions_map_path=positions_map_path)
            with open(pos_dict_path, 'wb') as f:
                pickle.dump(pos_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        STG = SpatioTemporalGraphCVAE(pos_dict, robot_node, 
                                      edge_radius=EDGE_RADIUS,
                                      edge_state_combine_method=EDGE_STATE_COMBINE_METHOD,
                                      edge_influence_combine_method=EDGE_INFLUENCE_COMBINE_METHOD)

        train_data_dict_path = os.path.join(config['data_dir'], "train_data_dict_%d_files_%s_rows.pkl" % (NUM_DATAFILES, str(ROWS_TO_EXTRACT)))
        if os.path.isfile(train_data_dict_path):
            with open(train_data_dict_path, 'rb') as f:
                train_data_dict = pickle.load(f)
        else:
            train_data_dict = get_data_dict(train_files, positions_map_path=positions_map_path)
            with open(train_data_dict_path, 'wb') as f:
                pickle.dump(train_data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        # save_as_npz_and_hdf5(base_filename + "_train", train_data_dict["input_dict"])

        hps.add_hparam("nodes_standardization", train_data_dict["nodes_standardization"])
        hps.add_hparam("extras_standardization", {"mean": train_data_dict["extras_mean"],
                                                  "std": train_data_dict["extras_std"]})
        hps.add_hparam("labels_standardization", train_data_dict["labels_standardization"])
        hps.add_hparam("pred_indices", train_data_dict["pred_indices"])

        eval_data_dict_path = os.path.join(config['data_dir'], "eval_data_dict_%d_files_%s_rows.pkl" % (NUM_DATAFILES, str(ROWS_TO_EXTRACT)))
        if os.path.isfile(eval_data_dict_path):
            with open(eval_data_dict_path, 'rb') as f:
                eval_data_dict = pickle.load(f)
        else:
            eval_data_dict = get_data_dict(eval_files, positions_map_path=positions_map_path)
            with open(eval_data_dict_path, 'wb') as f:
                pickle.dump(eval_data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        # save_as_npz_and_hdf5(base_filename + "_eval", eval_data_dict["input_dict"])

        train_input_function = tf.estimator.inputs.numpy_input_fn(train_data_dict["input_dict"],
                                                                  y = train_data_dict["labels"],
                                                                  batch_size = hps.batch_size,
                                                                  num_epochs = None,
                                                                  shuffle = True)

        # Need all possible nodes to have been seen by the STG above, does 
        # that mean we feed in the all_files pos_dict in order to create 
        # the nodes ahead of time?
        token_eval_node = None
        token_eval_label_node = None
        for node in eval_data_dict["input_dict"]:
            if isinstance(node, STGNode):
                token_eval_node = node
                token_eval_label_node = convert_to_label_node(node)
                break

        for node in train_data_dict["input_dict"]:
            if isinstance(node, STGNode):
                if node not in eval_data_dict["input_dict"]:
                    eval_data_dict["input_dict"][node] = np.zeros_like(eval_data_dict["input_dict"][token_eval_node])
                    eval_data_dict["labels"][convert_to_label_node(node)] = np.zeros_like(eval_data_dict["labels"][token_eval_label_node])

        eval_input_function = tf.estimator.inputs.numpy_input_fn(eval_data_dict["input_dict"],
                                                                 y = eval_data_dict["labels"],
                                                                 batch_size = 4,
                                                                 num_epochs = 1,
                                                                 shuffle = False)
        
        self.STG = STG
        self.hps = hps
        self.train_input_function = train_input_function
        self.eval_input_function = eval_input_function
        
        self.train_data_dict = train_data_dict
        logging.info('[load_data_and_define_model] Done!')


    def setup_model(self):
        train_input_function = self.train_input_function
        eval_input_function = self.eval_input_function
        
        laneswap_model = self.STG
        self.nn = nn = tf.estimator.Estimator(laneswap_model.model_fn, params=self.hps, 
                                              config=self.rc, model_dir=self.model_dir)
        self.experiment = tf.contrib.learn.Experiment(nn, train_input_function, eval_input_function,
                                                      eval_steps=None)
        logging.info('[setup_model] Done!')
        
    
#     @profile
    def train_and_evaluate(self):
        logging.info('[train_and_evaluate] Started!')
#         self.experiment.continuous_train_and_eval()
#         self.experiment.train_and_evaluate()
        self.experiment.train()
        logging.info('[train_and_evaluate] Done!')
            

    def print_num_params(self, level=2):
        nn = self.nn
        
        variable_names = nn.get_variable_names()
        if level == 0:
            # Total number of parameters
            num_params = np.sum([np.prod(nn.get_variable_value(var_name).shape) for var_name in variable_names]).astype(int)
            logging.info("Total number of parameters: {:,}".format(num_params))

        else:
            node_type_params = defaultdict(int)
            for variable_name in variable_names:
                key = '/'.join(variable_name.split('/')[:level])
                node_type_params[key] += np.prod(nn.get_variable_value(variable_name).shape).astype(int)

            for (key, value) in node_type_params.iteritems():
                logging.info("{}: {:,}".format(key, value))

        logging.info("-"*40)

    
#     @profile
    def save_model(self):
        nodes = [node for node in self.train_data_dict["input_dict"] if isinstance(node, STGNode)]
        
        state_dim = self.train_data_dict["input_dict"][nodes[0]].shape[2]
        extras_dim = self.train_data_dict["input_dict"]["extras"].shape[2]
        ph = self.hps.prediction_horizon

        with tf.Graph().as_default():
            input_dict = {"extras": tf.placeholder(tf.float32, shape=[1, None, extras_dim], name="extras"),
                          "sample_ct": tf.placeholder(tf.int32, shape=[1], name="sample_ct"),
                          "traj_lengths": tf.placeholder(tf.int32, shape=[1], name="traj_lengths")}
            
            for node in nodes:
                input_dict[str(node)] = tf.placeholder(tf.float32, shape=[1, None, state_dim], name=str(node))
                
#             input_dict[str(self.robot_node) + "_future_x"] = tf.placeholder(tf.float32, 
#                                                                             shape=[None, ph, state_dim/2], 
#                                                                             name=str(self.robot_node) + "_future_x")
#             input_dict[str(self.robot_node) + "_future_y"] = tf.placeholder(tf.float32, 
#                                                                             shape=[None, ph, state_dim/2], 
#                                                                             name=str(self.robot_node) + "_future_y")
            input_dict[str(self.robot_node) + "_future"]   = tf.placeholder(tf.float32, 
                                                                            shape=[None, ph, state_dim], 
                                                                            name=str(self.robot_node) + "_future")
            
            serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(input_dict)
            save_path = self.nn.export_savedmodel(config["models_dir"], serving_input_receiver_fn)
        
        logging.info('[save_model] Done! Saved to %s' % save_path)
    
    
    def run(self):
        self.setup()
        self.load_data_and_define_model()
        self.setup_model()
                
        self.train_and_evaluate()
        
        self.print_num_params(level=0)
        self.print_num_params(level=1)
        self.print_num_params(level=2)
        
        self.save_model()


def main():
    runner = Runner()
    runner.run()
    
    
if __name__ == "__main__":
    main()
