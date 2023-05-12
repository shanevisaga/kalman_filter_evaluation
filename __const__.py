###############################################
#Change values here############################
###############################################
import pytz
import numpy as np

import glob
import os

import pandas as pd
import csv

import matplotlib.dates as md
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

import datetime
import math

from tqdm import tqdm

from scipy import stats
from scipy.stats import pearsonr
from scipy.stats import kendalltau
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from numpy import dot
from numpy.linalg import inv


station_name = 'MO'

tz = pytz.timezone("Asia/Manila")
resolution = '10Min'

idx_t = 0
recursive_calculation_covariance_matrices = True
add_noise_in_predictions = False
nonlinear_predictions = False
predictors_vector = ['Kc_GHI_pred', 'cossza' ]
#predictors_vector = ['Kc_GHI_pred', 'cossza', 'U10', 'V10', 'T2', 'Humi_rel'  ]
#PREDICTORS = ['U10', 'V10', 'CC_b', 'CC_m', 'CC_t', 'T2', 'Q2_rel', 'cosSZA']

seas = 'janmar'
main_dir_path_recursive = f'output_{seas}'

if seas == 'janmar':
    nbhd = 42 #number of training days
elif seas == 'junaug':
    nbhd = 30 #number of training days
