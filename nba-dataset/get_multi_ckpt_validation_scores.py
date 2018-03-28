import sys
from os.path import exists, join
sys.path.append("..")
from experiment_details import get_output_ckpts_dir_name

from subprocess import call


def get_valid_score(experiment_name, ckpt_path, source_str='source activate tensorflow_p27'):
    call('screen -S %s -dm bash -c "%s; python get_ckpt_validation_score.py %s; exec sh"' % (experiment_name, 
                                                                                             source_str, 
                                                                                             ckpt_path), shell=True)
    

ckpt_num = 231
dir_prefix = 'models'

screen_names = list()

radii = ['2_files_100_rows_3.280840_edge_radius_mean_edge_inputs_bi-rnn_influences_ckpts',
         '2_files_100_rows_6.561680_edge_radius_mean_edge_inputs_bi-rnn_influences_ckpts',
         '2_files_100_rows_9.842520_edge_radius_mean_edge_inputs_bi-rnn_influences_ckpts',
         '2_files_100_rows_13.123360_edge_radius_mean_edge_inputs_bi-rnn_influences_ckpts',
         '2_files_100_rows_16.404200_edge_radius_mean_edge_inputs_bi-rnn_influences_ckpts']
radii_names = [name.split('_')[4] + 'ft_radius_valid' for name in radii]

# Get the bi-rnn one from the 3m radius one above!
influences = ['2_files_100_rows_6.561680_edge_radius_mean_edge_inputs_max_influences_ckpts',
              '2_files_100_rows_6.561680_edge_radius_mean_edge_inputs_sum_influences_ckpts']
influence_names = [name.split('_')[10] + '_influence_valid' for name in influences]

# Get the sum one from the 3m radius one above!
inputs = ['2_files_100_rows_6.561680_edge_radius_sum_edge_inputs_bi-rnn_influences_ckpts']
input_names = [name.split('_')[7] + '_input_valid' for name in inputs]

dir_list = radii + influences + inputs
screen_names = radii_names + influence_names + input_names

for i, model in enumerate(dir_list):
    get_valid_score(screen_names[i], join(dir_prefix, model, 'model.ckpt-%d' % ckpt_num))
