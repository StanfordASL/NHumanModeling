from __future__ import absolute_import, division, print_function
import timeit
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.DEBUG)
import matplotlib.pyplot as plt
import random
import shutil
from collections import defaultdict

import cPickle as pickle

import sys
sys.path.append("../code")
from st_graph import hps
from stg_node import STGNode
from experiment_details import get_output_base

robot_stg_node = STGNode('Mason Plumlee', 'HomeC')
robot_node = str(robot_stg_node)

from glob import glob
from st_graph import *
from data_utils import *
from stg_node import *

tf.reset_default_graph()

if len(sys.argv) < 7:
    print('Usage: source activate tensorflow_p27; python get_single_scalability_info.py <NUM_DATAFILES> <ROWS_TO_EXTRACT> <EDGE_RADIUS> <EDGE_SCM> <EDGE_ICM> <ROWS_TO_USE>')

NUM_DATAFILES = int(sys.argv[1])
ROWS_TO_EXTRACT = sys.argv[2]
EDGE_RADIUS = float(sys.argv[3])
EDGE_SCM = sys.argv[4]
EDGE_ICM = sys.argv[5]
ROWS_TO_USE = sys.argv[6]

data_dir = "data/2016.NBA.Raw.SportVU.Game.Logs"
all_files = glob(os.path.join(data_dir, '*.json'))[:NUM_DATAFILES]
print(all_files)
train_files, eval_files, test_files = split_files(all_files, train_eval_test_split=[.8, .2, .0], seed=123)
print(train_files, eval_files)
positions_map_path = "data/positions_map.pkl"
pos_dict_path = "data/pos_dict_%d_files_%s_rows.pkl" % (NUM_DATAFILES, str(ROWS_TO_EXTRACT))

if os.path.isfile(pos_dict_path):
    with open(pos_dict_path, 'rb') as f:
        pos_dict = pickle.load(f)
else:
    pos_dict = get_pos_dict(train_files, positions_map_path=positions_map_path)
    with open(pos_dict_path, 'wb') as f:
        pickle.dump(pos_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

STG = SpatioTemporalGraphCVAE(pos_dict, robot_stg_node,
                              edge_radius=EDGE_RADIUS,
                              edge_state_combine_method=EDGE_SCM,
                              edge_influence_combine_method=EDGE_ICM)

train_data_dict_path = "data/train_data_dict_%d_files_%s_rows.pkl" % (NUM_DATAFILES, str(ROWS_TO_EXTRACT))
if os.path.isfile(train_data_dict_path):
    with open(train_data_dict_path, 'rb') as f:
        train_data_dict = pickle.load(f)
else:
    train_data_dict = get_data_dict(train_files, positions_map_path=positions_map_path)
    with open(train_data_dict_path, 'wb') as f:
        pickle.dump(train_data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

if ROWS_TO_USE != "all":
    ROWS_TO_USE = int(ROWS_TO_USE)
    for key in train_data_dict["input_dict"]:
        train_data_dict["input_dict"][key] = train_data_dict["input_dict"][key][:ROWS_TO_USE]
    for key in train_data_dict["labels"]:
        train_data_dict["labels"][key] = train_data_dict["labels"][key][:ROWS_TO_USE]

hps.add_hparam("nodes_standardization", train_data_dict["nodes_standardization"])
hps.add_hparam("extras_standardization", {"mean": train_data_dict["extras_mean"],
                                          "std": train_data_dict["extras_std"]})
hps.add_hparam("labels_standardization", train_data_dict["labels_standardization"])
hps.add_hparam("pred_indices", train_data_dict["pred_indices"])
        
eval_input_function = tf.estimator.inputs.numpy_input_fn(train_data_dict["input_dict"],
                                                         y = train_data_dict["labels"],
                                                         batch_size = 4,
                                                         num_epochs = 1,
                                                         shuffle = False)

mode = tf.estimator.ModeKeys.PREDICT
features, labels = eval_input_function()
model_dir = '.'

sc = tf.ConfigProto(device_count={'GPU': 1},
                    allow_soft_placement=True)
rc = tf.estimator.RunConfig().replace(session_config=sc, 
                                      model_dir=model_dir,
                                      save_summary_steps=10,
                                      keep_checkpoint_max=None,
                                      tf_random_seed=None)

nn_estimator = tf.estimator.Estimator(STG.model_fn, params=hps, 
                                      config=rc, model_dir=model_dir)

features = {str(k): v for k, v in features.iteritems()}
features['sample_ct'] = [100]
random_node = [node for node in features if '/' in node][0]
features[str(robot_stg_node)] = features[random_node]
features[str(robot_stg_node) + "_future"] = features[str(robot_stg_node)]
print(features.keys())
# Creating the actual model
nn = nn_estimator.model_fn(features=features,
                           labels=labels,
                           mode=mode, 
                           config=rc)

import os
import psutil
def get_memory_usage():
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
    return memoryUse

pre = get_memory_usage()
# Creating the actual model
nn = nn_estimator.model_fn(features=features,
                           labels=labels,
                           mode=mode, 
                           config=rc)
post = get_memory_usage()
mem_usage = post - pre

try:
    global_step = tf.train.create_global_step()
except:
    print("global_step already exists")

graph_vars_saver = tf.train.Saver()

sess = tf.Session()

sess.run(tf.global_variables_initializer())

save_path = graph_vars_saver.save(sess, os.path.join(model_dir, 'model_%d_files_%s_rows.ckpt' % (NUM_DATAFILES, 
                                                                                                 str(ROWS_TO_EXTRACT))))
print("Saved model to %s" % save_path)

def save_predictive_model(train_data_dict, nn_estimator, models_dir):
    nodes = [node for node in train_data_dict['input_dict'] if isinstance(node, STGNode)]

    state_dim = train_data_dict['input_dict'][nodes[0]].shape[2]
    extras_dim = train_data_dict['input_dict']["extras"].shape[2]
    ph = hps.prediction_horizon

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
        input_dict[str(robot_stg_node) + "_future"]   = tf.placeholder(tf.float32, 
                                                                       shape=[None, ph, state_dim], 
                                                                       name=str(robot_stg_node) + "_future")

        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(input_dict)
        save_path = nn_estimator.export_savedmodel(models_dir, serving_input_receiver_fn)
        
    return save_path

saved_eval_predictive_model = save_predictive_model(train_data_dict, nn_estimator, model_dir)

total_parameters = 0
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim.value
#     print(variable.name, shape, '=>', variable_parameters)
    total_parameters += variable_parameters

num_nodes = len(STG.node_edges_and_neighbors)

num_edges = sum(len(value) for _, value in STG.node_edges_and_neighbors.iteritems())/2

tf.reset_default_graph()

with tf.Graph().as_default() as g:
    sess = tf.Session()
    print("Loading model from: " + saved_eval_predictive_model)
    tf.saved_model.loader.load(sess,
                               [tf.saved_model.tag_constants.SERVING],
                               saved_eval_predictive_model)

predict_data_dict = {str(k): v for k, v in train_data_dict['input_dict'].iteritems()}

data_id = 0
t_predict = 10
predict_horizon = 15
num_samples = 100

tic = timeit.default_timer()

tic0 = timeit.default_timer()
feed_dict = {k + ':0': v[[data_id]] for k, v in predict_data_dict.iteritems() if '/' in k}

robot_future = predict_data_dict[robot_node][data_id : data_id+1, t_predict + 1 : t_predict + predict_horizon + 1]
feed_dict[robot_node + "_future:0"] = robot_future

feed_dict["traj_lengths:0"] = [t_predict]
feed_dict["sample_ct:0"] = [num_samples]
feed_dict["extras:0"] = predict_data_dict["extras"][[data_id]]
toc0 = timeit.default_timer()

print("constructing feed_dict took: ", toc0 - tic0, " (s), running tf!")

run_list = list()
for node_str in predict_data_dict:
    if '/' not in node_str or robot_node in node_str:
        continue

    run_list.extend([node_str + '_1/outputs/y:0',
                     node_str + '_1/outputs/z:0'])

runtimes = list()
num_runs = 100
for _ in xrange(num_runs):
    tic0 = timeit.default_timer()
    outputs = sess.run(run_list, feed_dict=feed_dict)
    toc0 = timeit.default_timer()
    
    runtimes.append(toc0 - tic0)

avg_runtime = np.mean(runtimes)

print(get_output_base(NUM_DATAFILES, ROWS_TO_EXTRACT, 
                      EDGE_RADIUS, EDGE_SCM, EDGE_ICM))
print('-'*20)
print('Num Nodes:', num_nodes)
print('Num Edges:', num_edges)
print('Num Params:', total_parameters)
print('GB Used:', mem_usage)
print('Avg runtime of %d forward passes:' % num_runs, avg_runtime)
