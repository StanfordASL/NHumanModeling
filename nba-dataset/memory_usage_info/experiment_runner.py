import sys
from os.path import exists, join
sys.path.append("..")
from experiment_details import get_output_ckpts_dir_name

from subprocess import call

NUM_DATAFILES = 2
ROWS_TO_EXTRACT = 100
EDGE_RADIUS = 2.0 * 3.28084

def run_experiment(experiment_name, experiment, 
                   source_str='source activate tensorflow_p27',
                   rerun=False):
    name_relevant_params = experiment[1:]
    if rerun or not exists(join('..', 'models', get_output_ckpts_dir_name(*name_relevant_params))):
        (gpu, num_datafiles, num_rows, edge_radius, edge_state_combine_method, edge_influence_combine_method) = experiment
        print('Starting Experiment %s on GPU%d with %d files, %s rows, %.6f edge radius, "%s" edge combine method, "%s" edge influence combine method.' % (experiment_name,
                                                                                                                 gpu,
                                                                                                                 num_datafiles,
                                                                                                                 str(num_rows),
                                                                                                                 edge_radius,
                                                                                                                 edge_state_combine_method, edge_influence_combine_method))
    
        call('screen -S %s -dm bash -c "%s; python training_memory_usage.py %d %d %s %.6f %s %s; exec sh"' % (
                                                                                           experiment_name, 
                                                                                           source_str, 
                                                                                           gpu,
                                                                                           num_datafiles,
                                                                                           str(num_rows),
                                                                                           edge_radius,
                                                                                           edge_state_combine_method,
                                                                                           edge_influence_combine_method), 
             shell=True)

###############################
### Edge Radius Experiments ###
###############################
# Experiment 1 ----------------------------------------------------------------------------------------------------------------
experiment = (0,                # GPU
              NUM_DATAFILES,    # NUM_DATAFILES
              ROWS_TO_EXTRACT,  # ROWS_TO_EXTRACT
              1.0 * 3.28084,    # EDGE_RADIUS
              'mean',           # EDGE_STATE_COMBINE_METHOD
              'bi-rnn')         # EDGE_INFLUENCE_COMBINE_METHOD
run_experiment('1m_radius', experiment)
# -----------------------------------------------------------------------------------------------------------------------------

# Experiment 2 ----------------------------------------------------------------------------------------------------------------
experiment = (1,                # GPU
              NUM_DATAFILES,    # NUM_DATAFILES
              ROWS_TO_EXTRACT,  # ROWS_TO_EXTRACT
              2.0 * 3.28084,    # EDGE_RADIUS
              'mean',           # EDGE_STATE_COMBINE_METHOD
              'bi-rnn')         # EDGE_INFLUENCE_COMBINE_METHOD
run_experiment('2m_radius', experiment)
# -----------------------------------------------------------------------------------------------------------------------------

# Experiment 3a ---------------------------------------------------------------------------------------------------------------
experiment = (2,                # GPU
              NUM_DATAFILES,    # NUM_DATAFILES
              ROWS_TO_EXTRACT,  # ROWS_TO_EXTRACT
              3.0 * 3.28084,    # EDGE_RADIUS
              'mean',           # EDGE_STATE_COMBINE_METHOD
              'bi-rnn')         # EDGE_INFLUENCE_COMBINE_METHOD
run_experiment('3m_radius', experiment)
# -----------------------------------------------------------------------------------------------------------------------------

# Experiment 3b ---------------------------------------------------------------------------------------------------------------
experiment = (3,                # GPU
              NUM_DATAFILES,    # NUM_DATAFILES
              ROWS_TO_EXTRACT,  # ROWS_TO_EXTRACT
              4.0 * 3.28084,    # EDGE_RADIUS
              'mean',           # EDGE_STATE_COMBINE_METHOD
              'bi-rnn')         # EDGE_INFLUENCE_COMBINE_METHOD
run_experiment('4m_radius', experiment)
# -----------------------------------------------------------------------------------------------------------------------------

# Experiment 4 ----------------------------------------------------------------------------------------------------------------
experiment = (4,                # GPU
              NUM_DATAFILES,    # NUM_DATAFILES
              ROWS_TO_EXTRACT,  # ROWS_TO_EXTRACT
              5.0 * 3.28084,    # EDGE_RADIUS
              'mean',           # EDGE_STATE_COMBINE_METHOD
              'bi-rnn')         # EDGE_INFLUENCE_COMBINE_METHOD
run_experiment('5m_radius', experiment)
# -----------------------------------------------------------------------------------------------------------------------------

# Experiment 5 ----------------------------------------------------------------------------------------------------------------
# experiment = (5,                # GPU
#               NUM_DATAFILES,    # NUM_DATAFILES
#               ROWS_TO_EXTRACT,  # ROWS_TO_EXTRACT
#               6.0 * 3.28084,    # EDGE_RADIUS
#               'mean',           # EDGE_STATE_COMBINE_METHOD
#               'bi-rnn')         # EDGE_INFLUENCE_COMBINE_METHOD
# run_experiment('6m_radius', experiment)
# -----------------------------------------------------------------------------------------------------------------------------

# Experiment 6 ----------------------------------------------------------------------------------------------------------------
# experiment = (6,                # GPU
#               NUM_DATAFILES,    # NUM_DATAFILES
#               ROWS_TO_EXTRACT,  # ROWS_TO_EXTRACT
#               7.0 * 3.28084,    # EDGE_RADIUS
#               'mean',           # EDGE_STATE_COMBINE_METHOD
#               'bi-rnn')         # EDGE_INFLUENCE_COMBINE_METHOD
# run_experiment('7m_radius', experiment)
# -----------------------------------------------------------------------------------------------------------------------------

##########################################
### Edge Influence Combine Experiments ###
##########################################
# Experiment 1 ----------------------------------------------------------------------------------------------------------------
experiment = (5,                # GPU
              NUM_DATAFILES,    # NUM_DATAFILES
              ROWS_TO_EXTRACT,  # ROWS_TO_EXTRACT
              EDGE_RADIUS,      # EDGE_RADIUS
              'mean',           # EDGE_STATE_COMBINE_METHOD
              'sum')            # EDGE_INFLUENCE_COMBINE_METHOD
run_experiment('sum_edge_influence_combine', experiment)
# -----------------------------------------------------------------------------------------------------------------------------

# Experiment 2 ----------------------------------------------------------------------------------------------------------------
experiment = (6,                # GPU
              NUM_DATAFILES,    # NUM_DATAFILES
              ROWS_TO_EXTRACT,  # ROWS_TO_EXTRACT
              EDGE_RADIUS,      # EDGE_RADIUS
              'mean',           # EDGE_STATE_COMBINE_METHOD
              'max')            # EDGE_INFLUENCE_COMBINE_METHOD
run_experiment('max_edge_influence_combine', experiment)
# -----------------------------------------------------------------------------------------------------------------------------


######################################
### Edge Input Combine Experiments ###
######################################
# Experiment 1 ----------------------------------------------------------------------------------------------------------------
# experiment = (0,                # GPU
#               NUM_DATAFILES,    # NUM_DATAFILES
#               ROWS_TO_EXTRACT,  # ROWS_TO_EXTRACT
#               EDGE_RADIUS,      # EDGE_RADIUS
#               'max',            # EDGE_STATE_COMBINE_METHOD
#               'bi-rnn')         # EDGE_INFLUENCE_COMBINE_METHOD
# run_experiment('max_edge_input_combine', experiment)
# -----------------------------------------------------------------------------------------------------------------------------

# Experiment 2 ----------------------------------------------------------------------------------------------------------------
experiment = (7,                # GPU
              NUM_DATAFILES,    # NUM_DATAFILES
              ROWS_TO_EXTRACT,  # ROWS_TO_EXTRACT
              EDGE_RADIUS,      # EDGE_RADIUS
              'sum',            # EDGE_STATE_COMBINE_METHOD
              'bi-rnn')         # EDGE_INFLUENCE_COMBINE_METHOD
run_experiment('sum_edge_input_combine', experiment)
# -----------------------------------------------------------------------------------------------------------------------------
