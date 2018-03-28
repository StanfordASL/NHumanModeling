import sys
from os.path import exists, join
sys.path.append("..")
from experiment_details import get_output_ckpts_dir_name
import numpy as np
from subprocess import call


def get_scale_info(NUM_DATAFILES, ROWS_TO_EXTRACT, EDGE_RADIUS, EDGE_SCM, EDGE_ICM, ROWS_TO_USE, source_str='source activate tensorflow_p27'):
    experiment_name = 'SCALE_' + get_output_ckpts_dir_name(NUM_DATAFILES, ROWS_TO_EXTRACT, EDGE_RADIUS, EDGE_SCM, EDGE_ICM)
    call('screen -S %s -dm bash -c "%s; pip install psutil; python -u get_single_scalability_info.py %d %s %f %s %s %s > scalability_%d_%s.txt; exec sh"' % (experiment_name, 
                                                                                             source_str,
                                                                                             NUM_DATAFILES,
                                                                                             ROWS_TO_EXTRACT,
                                                                                             EDGE_RADIUS,
                                                                                             EDGE_SCM,
                                                                                             EDGE_ICM, ROWS_TO_USE, NUM_DATAFILES, ROWS_TO_EXTRACT), shell=True)

idx = int(sys.argv[1])
num_datafiles = range(2, 11)
rows_list = [2]*9
#rows_list = [200, 380, 560, 740, 920, 1100, 1280, 1460, 1640]
files_list = [(num_datafiles[idx], rows_list[idx], 'all')]
print('Looking at', files_list)

for (NUM_DATAFILES, ROWS_TO_EXTRACT, ROWS_TO_USE) in files_list:
    for EDGE_RADIUS in 3.28084 * np.arange(3.0, 3.1):
        for EDGE_SCM in ['mean']:
            for EDGE_ICM in ['bi-rnn']:
                get_scale_info(NUM_DATAFILES, ROWS_TO_EXTRACT, EDGE_RADIUS, EDGE_SCM, EDGE_ICM, ROWS_TO_USE)
