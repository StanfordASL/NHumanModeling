from __future__ import absolute_import, division, print_function
import timeit
import sys

if len(sys.argv) < 2:
    print('Usage: source activate tensorflow_p27; python plot_validation_curves.py <model checkpoint>')
    
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

checkpoint_file = sys.argv[1]
checkpoints_dir = '/'.join(checkpoint_file.split('/')[:-1])

import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.DEBUG)
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import random
import shutil
from collections import defaultdict
from scipy.integrate import cumtrapz
import cPickle as pickle
from experiment_details import extract_experiment_info, get_output_base

sys.path.append("../code")
from glob import glob
from st_graph import *
from data_utils import *
from stg_node import *
from utils.learning import _SUPER_SECRET_EVAL_KEY

NUM_DATAFILES, ROWS_TO_EXTRACT, EDGE_RADIUS, EDGE_STATE_COMBINE_METHOD, EDGE_INFLUENCE_COMBINE_METHOD = extract_experiment_info(checkpoints_dir)

output_base = get_output_base(NUM_DATAFILES, ROWS_TO_EXTRACT, EDGE_RADIUS, 
                              EDGE_STATE_COMBINE_METHOD, EDGE_INFLUENCE_COMBINE_METHOD)

robot_stg_node = STGNode('Al Horford', 'HomeC')
robot_node = str(robot_stg_node)

tf.reset_default_graph()

data_dir = "data/2016.NBA.Raw.SportVU.Game.Logs"
all_files = glob(os.path.join(data_dir, '*at_ATL*.json'))[:NUM_DATAFILES]
train_files, eval_files, test_files = split_files(all_files, train_eval_test_split=[.8, .2, .0], seed=123)

positions_map_path = "data/positions_map.pkl"
pos_dict_path = "data/pos_dict_eval_%d_files_%s_rows.pkl" % (NUM_DATAFILES, str(ROWS_TO_EXTRACT))

if os.path.isfile(pos_dict_path):
    with open(pos_dict_path, 'rb') as f:
        pos_dict = pickle.load(f)
else:
    pos_dict = get_pos_dict(eval_files, positions_map_path=positions_map_path)
    with open(pos_dict_path, 'wb') as f:
        pickle.dump(pos_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

STG = SpatioTemporalGraphCVAE(pos_dict, robot_stg_node,
                              edge_radius=EDGE_RADIUS,
                              edge_state_combine_method=EDGE_STATE_COMBINE_METHOD,
                              edge_influence_combine_method=EDGE_INFLUENCE_COMBINE_METHOD)

train_data_dict_path = "data/train_data_dict_%d_files_%s_rows.pkl" % (NUM_DATAFILES, str(ROWS_TO_EXTRACT))
if os.path.isfile(train_data_dict_path):
    with open(train_data_dict_path, 'rb') as f:
        train_data_dict = pickle.load(f)

eval_data_dict_path = "data/eval_data_dict_%d_files_%s_rows.pkl" % (NUM_DATAFILES, str(ROWS_TO_EXTRACT))
if os.path.isfile(eval_data_dict_path):
    with open(eval_data_dict_path, 'rb') as f:
        eval_data_dict = pickle.load(f)

hps.add_hparam("nodes_standardization", eval_data_dict["nodes_standardization"])
hps.add_hparam("extras_standardization", {"mean": eval_data_dict["extras_mean"],
                                          "std": eval_data_dict["extras_std"]})
hps.add_hparam("labels_standardization", eval_data_dict["labels_standardization"])
hps.add_hparam("pred_indices", eval_data_dict["pred_indices"])
        
eval_input_function = tf.estimator.inputs.numpy_input_fn(eval_data_dict["input_dict"],
                                                         y = eval_data_dict["labels"],
                                                         batch_size = 4,
                                                         num_epochs = 1,
                                                         shuffle = False)

mode = tf.estimator.ModeKeys.EVAL
features, labels = eval_input_function()
model_dir = 'models/eval_models/curr_model_' + output_base

sc = tf.ConfigProto(device_count={'GPU': 1},
                    allow_soft_placement=True)
rc = tf.estimator.RunConfig().replace(session_config=sc, 
                                      model_dir=model_dir,
                                      save_summary_steps=10,
                                      keep_checkpoint_max=None,
                                      tf_random_seed=None)

nn_estimator = tf.estimator.Estimator(STG.model_fn, params=hps, 
                                      config=rc, model_dir=model_dir)

# Creating the actual model
nn = nn_estimator.model_fn(features=features,
                           labels=labels,
                           mode=mode, 
                           config=rc)

def save_eval_model(train_data_dict, nn_estimator, models_dir):
    nodes = [node for node in train_data_dict['input_dict'] if isinstance(node, STGNode)]

    pred_dim = len(train_data_dict['pred_indices'])
    state_dim = train_data_dict['input_dict'][nodes[0]].shape[2]
    extras_dim = train_data_dict['input_dict']["extras"].shape[2]
    ph = hps.prediction_horizon

    with tf.Graph().as_default():
        input_dict = {_SUPER_SECRET_EVAL_KEY: tf.placeholder(tf.float32, shape=[1], name="NOT_FOR_USE"),
                      "bag_idx": tf.placeholder(tf.int32, shape=[1, None, 1], name="bag_idx"),
                      "extras": tf.placeholder(tf.float32, shape=[1, None, extras_dim], name="extras"),
                      "traj_lengths": tf.placeholder(tf.int32, shape=[1], name="traj_lengths")}

        for node in nodes:
            input_dict[str(node)] = tf.placeholder(tf.float32, shape=[1, None, state_dim], name=str(node))
            
            labels_node = convert_to_label_node(node)
            input_dict[str(labels_node)] = tf.placeholder(tf.float32, shape=[1, None, pred_dim], 
                                                          name=str(labels_node))

        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(input_dict)
        save_path = nn_estimator.export_savedmodel(models_dir, serving_input_receiver_fn)
        
    return save_path

# GETTING VALIDATION SCORES
from tensorflow.contrib.framework.python.framework import checkpoint_utils
import re
from collections import defaultdict

files = [checkpoint_file]
eval_data_dict = pickle.load(open('data/eval_data_dict_%d_files_%s_rows.pkl' % (NUM_DATAFILES, str(ROWS_TO_EXTRACT)), 'rb'))

random_model_dir = files[0]
ckpt_var_list = [var_name for (var_name, shape) in checkpoint_utils.list_variables(random_model_dir)]

vars_to_restore = list()
for graph_var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    if graph_var.name[:-2] in ckpt_var_list:
        vars_to_restore.append(graph_var)

try:
    global_step = tf.train.create_global_step()
except:
    print("global_step already exists")
    
if global_step not in vars_to_restore:
    vars_to_restore.append(global_step)

init_global = tf.global_variables_initializer()
init_local = tf.local_variables_initializer()
train_vars_loader = tf.train.Saver(vars_to_restore)
graph_vars_saver = tf.train.Saver()

val_dict = defaultdict(list)
for model_ckpt in files:
    print(model_ckpt)
    train_model_dir = model_ckpt
    
    # 1) Loading trained vars.
    # 2) Adding in our eval vars (in case there are some that aren't in training).
    # 3) Saving the overall eval graph.
    with tf.Session() as sess:
        # Initialize all variables so everything has a default value.
        sess.run(init_global)
        sess.run(init_local)

        # Restore variables from disk that we also have in our graph.
        train_vars_loader.restore(sess, train_model_dir)
        print("Parameters restored.")

        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        save_path = graph_vars_saver.save(sess, os.path.join(model_dir, 'model.ckpt'))
        print("Saved model to %s" % save_path)

#         graph_path = tf.train.write_graph(sess.graph_def, model_dir, 'graph.pbtxt')
#         print("Wrote graph to %s" % graph_path)
    
    saved_eval_eval_model = save_eval_model(eval_data_dict, nn_estimator, 'models/eval_models/eval_' + output_base)
    
    with tf.Graph().as_default() as g:
        sess = tf.Session()
        print("Loading model from: " + saved_eval_eval_model)
        tf.saved_model.loader.load(sess,
                                   [tf.saved_model.tag_constants.SERVING],
                                   saved_eval_eval_model)
        
    tic = timeit.default_timer()

    tic0 = timeit.default_timer()
    feed_dict = {str(k) + ':0': v for k, v in eval_data_dict['input_dict'].iteritems() if isinstance(k, STGNode)}
    feed_dict.update({str(k) + ':0': v for k, v in eval_data_dict['labels'].iteritems() if isinstance(k, STGNode)})

    feed_dict["traj_lengths:0"] = eval_data_dict['input_dict']['traj_lengths']
    feed_dict["bag_idx:0"] = eval_data_dict['input_dict']['bag_idx']
    feed_dict["extras:0"] = eval_data_dict['input_dict']['extras']
    toc0 = timeit.default_timer()

    for key in feed_dict:
        feed_dict[key] = feed_dict[key][:10]

    print("constructing feed_dict took: ", toc0 - tic0, " (s), running tf!")

    run_list = list()
#     for node in eval_data_dict['input_dict']:
#         node_str = str(node)
#         if '/' not in node_str or robot_node in node_str:
#             continue

#         run_list.extend([#node_str + '/NLL_q_IS:0',
#                          #node_str + '/NLL_p:0',
#                          node_str + '/NLL_exact:0'])

    run_list.extend([#'ST-Graph/NLL_q_IS:0', 
                     #'ST-Graph/NLL_p:0', 
                     'ST-Graph/NLL_exact:0'])

    tic0 = timeit.default_timer()
    outputs = sess.run(run_list, feed_dict=feed_dict)
    toc0 = timeit.default_timer()

    print("done running tf!, took (s): ", toc0 - tic0)
    toc = timeit.default_timer()
    print("total time taken (s): ", toc - tic)
    
    sess.close()
    
    output_dict = dict(zip(run_list, outputs))
    
    for key in output_dict:
        val_dict[key].append(output_dict[key])
    
    shutil.rmtree(saved_eval_eval_model)
    shutil.rmtree(model_dir)
    print(checkpoint_file)
    print()
    print(val_dict)
