###############################################
#Change values here############################
###############################################
import pytz
import numpy as np

year = '2023'

station_name = 'MO'

tz = pytz.timezone("Asia/Manila")
resolution = '10Min'

idx_t = 0
recursive_calculation_covariance_matrices = True
add_noise_in_predictions = False
nonlinear_predictions = False
predictors_vector = ['Kc_GHI_pred', 'cossza', 'U10', 'V10', 'T2', 'Humi_rel'  ]
#PREDICTORS = ['U10', 'V10', 'CC_b', 'CC_m', 'CC_t', 'T2', 'Q2_rel', 'cosSZA']
main_dir_path_recursive = 'output'

