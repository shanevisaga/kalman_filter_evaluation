###############################################
#Change values here############################
###############################################
import pytz
import numpy as np

year = '2023'
ens = 'ensmean'
domain = 'd01'
station_name = 'MO'

tz = pytz.timezone("Asia/Manila")
resolution = '10Min'

idx_t = 0
recursive_calculation_covariance_matrices = True
add_noise_in_predictions = False
nonlinear_predictions = False
predictors_vector = ['Kc_GHI_pred', 'cossza']

main_dir_path_recursive = 'output'

