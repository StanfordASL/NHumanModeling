import os

NUM_DATAFILES = 2
ROWS_TO_EXTRACT = 100
EDGE_RADIUS = 2.0 * 3.28084 # This is a radius of 2m = approx 6.6ft
EDGE_STATE_COMBINE_METHOD = 'sum'
EDGE_INFLUENCE_COMBINE_METHOD = 'bi-rnn'

def get_output_base(num_datafiles, rows_to_extract, edge_radius, 
                    edge_state_combine_method, edge_influence_combine_method):
    return '%d_files_%s_rows_%.6f_edge_radius_%s_edge_inputs_%s_influences' % (num_datafiles, 
                                                                               str(rows_to_extract), 
                                                                               edge_radius, 
                                                                               edge_state_combine_method,
                                                                               edge_influence_combine_method)

def get_output_ckpts_dir_name(num_datafiles, rows_to_extract, edge_radius, 
                              edge_state_combine_method, edge_influence_combine_method):
    return get_output_base(num_datafiles, rows_to_extract, edge_radius, edge_state_combine_method, edge_influence_combine_method) + '_ckpts'

def extract_experiment_info(model_checkpoints_dir):
    ckpts_dir_str = os.path.basename(os.path.normpath(model_checkpoints_dir))
    split_str = ckpts_dir_str.split('_')
    num_files = int(split_str[0])
    num_rows = split_str[2]
    edge_radius = float(split_str[4])
    edge_state_combine_method = split_str[7]
    edge_influence_combine_method = split_str[10]
    
    return num_files, num_rows, edge_radius, edge_state_combine_method, edge_influence_combine_method
