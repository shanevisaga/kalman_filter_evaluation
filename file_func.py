from __const__ import *

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

def prep_opt(mod_, a):
    
    mod_d01 = mod_[(mod_['ens']== ens) & (mod_['domain']== domain) & (mod_['station_name']== station_name)]
    mod_d02 = mod_[(mod_['ens']== ens) & (mod_['domain']== domain) & (mod_['station_name']== station_name)]
    
    mod_d01  = mod_d01.set_index('Time')
    mod_d01  = mod_d01.resample(resolution).interpolate(method='linear')
    mod_d01  = mod_d01.reset_index(drop = False)
    mod_d01['station_name']  = station_name
    mod_d01['domain']  = domain
    mod_d01['ens']  = ens
    
    df_prep_opt = a.merge(mod_d01, how='inner', on='Time')
    df_prep_opt['Time'] = pd.to_datetime(df_prep_opt['Time']).dt.tz_convert(tz)

    df_prep_opt = df_prep_opt.set_index('Time')
    df_prep_opt['YY'] = pd.DatetimeIndex(df_prep_opt.index).year
    df_prep_opt['MM'] = pd.DatetimeIndex(df_prep_opt.index).month
    df_prep_opt['DD'] = pd.DatetimeIndex(df_prep_opt.index).day
    df_prep_opt['HH'] = pd.DatetimeIndex(df_prep_opt.index).hour
    df_prep_opt['mm'] = pd.DatetimeIndex(df_prep_opt.index).minute
    df_prep_opt = df_prep_opt.reset_index()
    
    df = df_prep_opt
    if 'Error_rel' not in df.columns:
        # Function to calculate relative errors
        def calculate_relative_errors(pred, obs):
            if obs == 0:
                return np.nan
            else:
                return (pred - obs)/obs
        # recalculate relative error after removing some observations
        df['Error_rel'] = df.apply(lambda row : calculate_relative_errors(row['ghi_mod'], row['ghi_obs']), axis = 1)

    #line 221
    df['Kc_GHI_pred'] = df['ghi_mod']/df['GHI_in']
    df['Kc_GHI_pred'] = df['Kc_GHI_pred'].replace(np.inf,np.nan)

    ##############################
    #CHECK THIS###################
    ##############################
    df['Kc_GHI_obs'] = df['ghi_obs']/df['GHI_in']

    #line 223
    # create clear-sky index column for observed errors 
    df['Kc_obs_bias'] = df['Kc_GHI_pred'] - df['Kc_GHI_obs']

    df['LT'] = (df['HH']*60)+df['mm']
    
    # ----------------------------
    #    Remove early morning and late afternoon
    # ----------------------------
    df_temp = df.loc[(df['Time'].dt.hour >= 8) & (df['Time'].dt.hour <= 17)].copy()
    idx_remove = df_temp.loc[(df_temp['Time'].dt.hour == 17) & (df_temp['Time'].dt.minute != 0)].index
    df_temp.drop(idx_remove, axis=0, inplace=True)
    df_temp = df_temp.reset_index(drop = False)
    
    #for predictors_vector in [['Kc_GHI_pred','Q2_rel']]:# [results_best_MBE.loc[results_best_MBE.index[-1], 'Predictors'].split('-')[1:], results_best_MAE.loc[results_best_MAE.index[-1], 'Predictors'].split('-')[1:]]:
    #df_temp = df.copy()
    df_temp['Kc_GHI_pred_improved'] = np.nan 

    ##############################
    #CHECK THIS###################
    ##############################
    df_temp['predicted_coefs'] = np.nan 
    df_temp =  df_temp[['Time', 'LT','CMP22_Total_Solar', 'SPN1_Total_Solar',
           'SPN1_Diff_Solar', 'CGR4_IR', 'YY', 'MM', 'HH', 'mm', 'dhi',
           'ghi_obs', 'sza', 'cossza', 'dni',  'GHI_in', 'DNI_in', 'DHI_in', 'cossza_b',
           'SPN1_Total_Solar_N', 't2_lim', 'cossza_noon', 'FT_t', 'FT_TOA',
           'FT_TOA_t', 't3_llim', 't3_ulim', 'Diffuse_Ratio', 'SPN1_Diff_Solar_N',
           'sigma', 'ghi_cc_val', 'dhi_cc_val', 't1_lim', 'flag_clear',\
             'ens', 'domain','station_name', 'ghi_mod', 'Error_rel', 'Kc_GHI_pred', 'Kc_GHI_obs', 'Kc_obs_bias',
           'Kc_GHI_pred_improved']]
    df_temp
    return(df,df_temp)
    
    
    
def fitting(da):

    x_new = np.linspace(0.01, 1360, 1000)
    x_a=np.array(da['ghi_obs'])
    y_a=np.array(da['ghi_mod'])
    xaa = x_a[~np.isnan(x_a)]
    yaa = y_a[~np.isnan(y_a)]
    slope_a, intercept_a, r_value_a, p_value_a, std_err_a = stats.linregress(x_a,y_a)
    pearson_corr, pearson_pval = pearsonr(xaa,yaa)
    coef, p = kendalltau(xaa,yaa)
    #print('Kendall correlation coefficient: %.3f' % coef)
    rms = mean_squared_error(xaa, yaa, squared=False)
    mae = mean_absolute_error(xaa, yaa)
    alpha = 0.01
    #if p > alpha:
        #print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
    #else:
        #print('Samples are correlated (reject H0) p=%.3f' % p)

    #print('Pearsons correlation: %.3f' % pearson_corr)
    #print('Pearsons p_value: %.3f' % pearson_pval)
    #print('RMS: %.3f' % rms)
    #print('MAE: %.3f' % mae)
    #print('###########################################')
    return (pearson_corr, pearson_pval, slope_a, intercept_a, r_value_a, p_value_a, std_err_a, rms, mae)

def fitting_pred(da):

    x_new = np.linspace(0.01, 1360, 1000)
    x_a=np.array(da['ghi_obs'])
    y_a=np.array(da['GHI_pred_improved_fin'])
    xaa = x_a[~np.isnan(x_a)]
    yaa = y_a[~np.isnan(y_a)]

    slope_a, intercept_a, r_value_a, p_value_a, std_err_a = stats.linregress(x_a,y_a)
    pearson_corr, pearson_pval = pearsonr(xaa,yaa)
    coef, p = kendalltau(xaa,yaa)
    #print('Kendall correlation coefficient: %.3f' % coef)
    rms = mean_squared_error(xaa, yaa, squared=False)
    mae = mean_absolute_error(xaa, yaa)
    
    alpha = 0.01
    #if p > alpha:
        #print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
    #else:
        #print('Samples are correlated (reject H0) p=%.3f' % p)

    #print('Pearsons correlation: %.3f' % pearson_corr)
    #print('RMS: %.3f' % rms)
    #print('MAE: %.3f' % mae)
    #print('###########################################')
    
    return (pearson_corr, pearson_pval, slope_a, intercept_a, r_value_a, p_value_a, std_err_a, rms, mae)
###########################

def kf_fitting(da):

    x_new = np.linspace(0.01, 1360, 1000)
    x_a=np.array(da['ghi_obs'])
    y_a=np.array(da['GHI_pred_kf_only'])
    xaa = x_a[~np.isnan(x_a)]
    yaa = y_a[~np.isnan(y_a)]
    slope_a, intercept_a, r_value_a, p_value_a, std_err_a = stats.linregress(x_a,y_a)
    pearson_corr, pearson_pval = pearsonr(xaa,yaa)
    coef, p = kendalltau(xaa,yaa)
    #print('Kendall correlation coefficient: %.3f' % coef)
    rms = mean_squared_error(xaa, yaa, squared=False)
    mae = mean_absolute_error(xaa, yaa)
    alpha = 0.01
    #if p > alpha:
        #print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
    #else:
        #print('Samples are correlated (reject H0) p=%.3f' % p)

    #print('Pearsons correlation: %.3f' % pearson_corr)
    #print('Pearsons p_value: %.3f' % pearson_pval)
    #print('RMS: %.3f' % rms)
    #print('MAE: %.3f' % mae)
    #print('###########################################')
    return (pearson_corr, pearson_pval, slope_a, intercept_a, r_value_a, p_value_a, std_err_a, rms, mae)

def kf_fitting_pred(da):

    x_new = np.linspace(0.01, 1360, 1000)
    x_a=np.array(da['ghi_obs'])
    y_a=np.array(da['GHI_pred_kf_only'])
    xaa = x_a[~np.isnan(x_a)]
    yaa = y_a[~np.isnan(y_a)]

    slope_a, intercept_a, r_value_a, p_value_a, std_err_a = stats.linregress(x_a,y_a)
    pearson_corr, pearson_pval = pearsonr(xaa,yaa)
    coef, p = kendalltau(xaa,yaa)
    #print('Kendall correlation coefficient: %.3f' % coef)
    rms = mean_squared_error(xaa, yaa, squared=False)
    mae = mean_absolute_error(xaa, yaa)
    
    alpha = 0.01
    #if p > alpha:
        #print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
    #else:
        #print('Samples are correlated (reject H0) p=%.3f' % p)

    #print('Pearsons correlation: %.3f' % pearson_corr)
    #print('RMS: %.3f' % rms)
    #print('MAE: %.3f' % mae)
    #print('###########################################')
    
    return (pearson_corr, pearson_pval, slope_a, intercept_a, r_value_a, p_value_a, std_err_a, rms, mae)

def ts_all(res, da_mean,dir_path_recursive):
    da_mean= da_mean
    fig = plt.figure()
    gs = fig.add_gridspec(6, 3)

    ax00 = fig.add_subplot(gs[0:1, 0:3])
    ax00.plot(da_mean.index,da_mean['ghi_obs'],c='DarkBlue',linewidth=1.5)
    ax00.plot(da_mean.index,da_mean['ghi_mod'],c='red',linewidth=1.)
    ax00.tick_params(axis='both', which='major', labelsize=5)
    ax00.set_ylim(0,1360)

    nb_historical_days = dir_path_recursive.split('/')[4]


    ax00.text(100, 1600, f'Manila Observatory WRF-Solar validation( {res} km | initialized 00 UTC previous day)\nYear: {year} | training period: {nb_historical_days}',  fontsize=5)
    #ax00.text(100, 1300, 'ensmean v obs',  fontsize=5)   
    ax00.text( 100, 1100,'WRF RMSE = %0.2f'%fitting(da_mean)[7], fontsize=5, color='red')
    ax00.legend(['Obs','WRF-' + str(res) + 'km'], bbox_to_anchor=(1.02, 0.98), prop={'size': 5},loc=2, borderaxespad=0.)
    ax00.set_xticklabels([])
    
    ax00 = fig.add_subplot(gs[2:3, 0:3])
    ax00.plot(da_mean.index,da_mean['ghi_obs'],c='DarkBlue',linewidth=1.5)
    ax00.plot(da_mean.index,da_mean['GHI_pred_improved_fin'],c='c',linewidth=1.0)
    ax00.tick_params(axis='both', which='major', labelsize=5)
    ax00.set_ylim(0,1360)
    ax00.text( 100, 1100,'KFv2-WRF RMSE = %0.2f'%fitting_pred(da_mean)[7], fontsize=5, color='c')
    ax00.legend(['Obs','KF+'], bbox_to_anchor=(1.02, 0.98), prop={'size': 5},loc=2, borderaxespad=0.)
    ax00.set_xticklabels([])
    
    ax00 = fig.add_subplot(gs[1:2, 0:3])
    ax00.plot(da_mean.index,da_mean['ghi_obs'],c='DarkBlue',linewidth=1.5)
    ax00.plot(da_mean.index,da_mean['GHI_pred_kf_only'],c='green',linewidth=1.0) 
    ax00.tick_params(axis='both', which='major', labelsize=5)
    ax00.set_ylim(0,1360)  
    ax00.text( 100, 1100,'KF-WRF RMSE = %0.2f'%kf_fitting_pred(da_mean)[7], fontsize=5, color='green')
    ax00.legend(['Obs','KF only'], bbox_to_anchor=(1.02, 0.98), prop={'size': 5},loc=2, borderaxespad=0.)
    ax00.set_xticklabels([])
    
    
    plt.savefig(f"{dir_path_recursive}/{year}_{nb_historical_days}_all_kf_Manila_ts_" + str(res) +"km.png", dpi=500, frameon=False, facecolor='white', bbox_inches="tight")
    
    
    with open(f'{dir_path_recursive}/{year}_{nb_historical_days}.csv', 'w') as fileObj:
        writerObj = csv.writer(fileObj)
        writerObj.writerow(['label','pearson_corr', 'rms', 'mae'])
        wrf = ('WRF all',fitting(da_mean)[0],fitting(da_mean)[7],fitting(da_mean)[8])
        kf = ('KF all',fitting_pred(da_mean)[0],fitting_pred(da_mean)[7],fitting_pred(da_mean)[8])
        writerObj.writerow(wrf)
        writerObj.writerow(kf) 
        fileObj.close()


############################
############################
############################
def ts_cut(res, da_mean):
    da_mean = da_mean.iloc[nb_LTs*nb_historical_days:,:] #

    fig = plt.figure()
    gs = fig.add_gridspec(6, 3)

    ax00 = fig.add_subplot(gs[0:1, 0:3])
    ax00.plot(da_mean.index,da_mean['ghi_obs'],c='DarkBlue',linewidth=1.5)
    ax00.plot(da_mean.index,da_mean['ghi_mod'],c='red',linewidth=1.)
    ax00.tick_params(axis='both', which='major', labelsize=5)
    ax00.set_ylim(0,1360)
    ax00.text(100, 1500, 'Manila Observatory WRF-Solar validation (' + str(res)+' km | initialized 00 UTC previous day)',  fontsize=5)
    #ax00.text(100, 1300, 'ensmean v obs',  fontsize=5)   
    ax00.text( 100, 1100,'WRF RMSE = %0.2f'%fitting(da_mean)[7], fontsize=5, color='red')
    ax00.legend(['Obs','WRF-' + str(res) + 'km'], bbox_to_anchor=(1.02, 0.98), prop={'size': 5},loc=2, borderaxespad=0.)
    ax00.set_xticklabels([])
    
    ax00 = fig.add_subplot(gs[2:3, 0:3])
    ax00.plot(da_mean.index,da_mean['ghi_obs'],c='DarkBlue',linewidth=1.5)
    ax00.plot(da_mean.index,da_mean['GHI_pred_improved_fin'],c='c',linewidth=1.0)
    ax00.tick_params(axis='both', which='major', labelsize=5)
    ax00.set_ylim(0,1360)
    ax00.text( 100, 1100,'KFv2-WRF RMSE = %0.2f'%fitting_pred(da_mean)[7], fontsize=5, color='c')
    ax00.legend(['Obs','KF+'], bbox_to_anchor=(1.02, 0.98), prop={'size': 5},loc=2, borderaxespad=0.)
    ax00.set_xticklabels([])
    
    ax00 = fig.add_subplot(gs[1:2, 0:3])
    ax00.plot(da_mean.index,da_mean['ghi_obs'],c='DarkBlue',linewidth=1.5)
    ax00.plot(da_mean.index,da_mean['GHI_pred_kf_only'],c='green',linewidth=1.0) 
    ax00.tick_params(axis='both', which='major', labelsize=5)
    ax00.set_ylim(0,1360)  
    ax00.text( 100, 1100,'KF-WRF RMSE = %0.2f'%kf_fitting_pred(da_mean)[7], fontsize=5, color='green')
    ax00.legend(['Obs','KF only'], bbox_to_anchor=(1.02, 0.98), prop={'size': 5},loc=2, borderaxespad=0.)
       

    plt.savefig(f"{images_dir}/{nb_historical_days}days/cut_kf_Manila_ts_" + str(res) +"km.png", dpi=500, frameon=False, facecolor='white', bbox_inches="tight")
    
    with open(f'{dir_path_recursive}/{year}_{nb_historical_days}.csv', 'a') as f_object:
        writerObj = csv.writer(f_object)
        wrf = ('WRF cut',fitting(da_mean)[0],fitting(da_mean)[7],fitting(da_mean)[8])
        kf = ('KF cut',fitting_pred(da_mean)[0],fitting_pred(da_mean)[7],fitting_pred(da_mean)[8])
        writerObj.writerow(wrf)
        writerObj.writerow(kf)
        f_object.close()


############################
############################
############################    
def scat_all(res, da_mean):   
    plt_da = da_mean
    x_new = np.linspace(0.01, 1360, 1000)
    fig = plt.figure()
    gs = fig.add_gridspec(4, 15)
    
    ax00 = fig.add_subplot(gs[0:1, 0:3])
    ax00.scatter(x=plt_da['ghi_obs'],y=plt_da['ghi_mod'],c='red',s=0.1,alpha=0.5)
    plt.plot(x_new,x_new,c='gray',linewidth=0.1)
    ax00.text( 50, 1000,'RMSE = %0.2f'%fitting(plt_da)[7], fontsize=3, color='red')
    ax00.text( 50, 1100,'R* = %0.2f'%fitting(plt_da)[0], fontsize=3, color='red')
    ax00.text( 200, 1390,'WRF-Solar '+ str(res) + '-km', fontsize=5, color='red')
    ax00.tick_params(axis='both', which='major', labelsize=5)
    ax00.set_ylim(0,1360)
    ax00.set_xlim(0,1360)
    ax00.set_ylabel('Model\nGHI (W/m$^2$)',  fontsize=5)   
    
    ax00 = fig.add_subplot(gs[0:1, 8:11])
    ax00.scatter(x=plt_da['ghi_obs'],y=plt_da['GHI_pred_improved_fin'],c='c',s=0.1,alpha=0.5)
    plt.plot(x_new,x_new,c='gray',linewidth=0.1)
    ax00.tick_params(axis='both', which='major', labelsize=5)
    ax00.text( 50, 1100,'R* = %0.2f'%fitting_pred(plt_da)[0], fontsize=3, color='c')
    ax00.text( 50, 1000,'RMSE = %0.2f'%fitting_pred(plt_da)[7], fontsize=3, color='c')  
    ax00.text( 200, 1390,'KF+ WRF-Solar '+ str(res) + '-km', fontsize=5, color='c')
    ax00.set_ylim(0,1360)
    ax00.set_xlim(0,1360)
    ax00.text( -500, -800,'Observed\nGHI (W/m$^2$)', fontsize=5, color='k')

        
    ax00 = fig.add_subplot(gs[0:1, 4:7])
    ax00.scatter(x=plt_da['ghi_obs'],y=plt_da['GHI_pred_kf_only'],c='green',s=0.1,alpha=0.6)
    plt.plot(x_new,x_new,c='gray',linewidth=0.1)
    ax00.tick_params(axis='both', which='major', labelsize=5)
    ax00.text( 50, 1100,'R* = %0.2f'%kf_fitting_pred(plt_da)[0], fontsize=3, color='green')
    ax00.text( 50, 1000,'RMSE = %0.2f'%kf_fitting_pred(plt_da)[7], fontsize=3, color='green')  
    ax00.text( 200, 1390,'KF WRF-Solar '+ str(res) + '-km', fontsize=5, color='green')
    ax00.set_ylim(0,1360)
    ax00.set_xlim(0,1360)
    ax00.text( -800, 1600,'All Periods', fontsize=5, color='k')
    
    images_dir = 'img/2023_runs'
    plt.savefig(f"{images_dir}/{nb_historical_days}days/all_kf_Manila_scatplot_" + str((res)) + "km.png", dpi=500, frameon=False, facecolor='white', bbox_inches="tight")


############################
############################
############################ 
def scat_cut(res, da_mean):   
    x_new = np.linspace(0.01, 1360, 1000)
    plt_da = da_mean.iloc[nb_LTs*nb_historical_days:,:] #
    
    fig = plt.figure()
    gs = fig.add_gridspec(4, 15) 
    
    ax00 = fig.add_subplot(gs[0:1, 0:3])
    ax00.scatter(x=plt_da['ghi_obs'],y=plt_da['ghi_mod'],c='red',s=0.1,alpha=0.5)
    plt.plot(x_new,x_new,c='gray',linewidth=0.1)
    ax00.text( 50, 1000,'RMSE = %0.2f'%fitting(plt_da)[7], fontsize=3, color='red')
    ax00.text( 50, 1100,'R* = %0.2f'%fitting(plt_da)[0], fontsize=3, color='red')
    ax00.text( 200, 1390,'WRF-Solar '+ str(res) + '-km', fontsize=5, color='red')
    ax00.tick_params(axis='both', which='major', labelsize=5)
    ax00.set_ylim(0,1360)
    ax00.set_xlim(0,1360)
    ax00.set_ylabel('Model\nGHI (W/m$^2$)',  fontsize=5)   
    
    
    ax00 = fig.add_subplot(gs[0:1, 8:11])
    ax00.scatter(x=plt_da['ghi_obs'],y=plt_da['GHI_pred_improved_fin'],c='c',s=0.1,alpha=0.5)
    plt.plot(x_new,x_new,c='gray',linewidth=0.1)
    ax00.tick_params(axis='both', which='major', labelsize=5)
    ax00.text( 50, 1100,'R* = %0.2f'%fitting_pred(plt_da)[0], fontsize=3, color='c')
    ax00.text( 50, 1000,'RMSE = %0.2f'%fitting_pred(plt_da)[7], fontsize=3, color='c')  
    ax00.text( 200, 1390,'KF+ WRF-Solar '+ str(res) + '-km', fontsize=5, color='c')
    ax00.set_ylim(0,1360)
    ax00.set_xlim(0,1360)
    ax00.text( -500, -800,'Observed\nGHI (W/m$^2$)', fontsize=5, color='k')

        
    ax00 = fig.add_subplot(gs[0:1, 4:7])
    ax00.scatter(x=plt_da['ghi_obs'],y=plt_da['GHI_pred_kf_only'],c='green',s=0.1,alpha=0.6)
    plt.plot(x_new,x_new,c='gray',linewidth=0.1)
    ax00.tick_params(axis='both', which='major', labelsize=5)
    ax00.text( 50, 1100,'R* = %0.2f'%kf_fitting_pred(plt_da)[0], fontsize=3, color='green')
    ax00.text( 50, 1000,'RMSE = %0.2f'%kf_fitting_pred(plt_da)[7], fontsize=3, color='green')  
    ax00.text( 200, 1390,'KF WRF-Solar '+ str(res) + '-km', fontsize=5, color='green')
    ax00.set_ylim(0,1360)
    ax00.set_xlim(0,1360)
    ax00.text( -800, 1600,'All Periods', fontsize=5, color='k')
    
    images_dir = 'img/2023_runs'
    plt.savefig(f"{images_dir}/{nb_historical_days}days/cut_kf_Manila_scatplot_" + str((res)) + "km.png", dpi=500, frameon=False, facecolor='white', bbox_inches="tight")


############################
############################
############################ 

def scat_cloudy_all(res, da_mean,dir_path_recursive):
    x_new = np.linspace(0.01, 1360, 1000)
    plt_da = da_mean
    plt_da = plt_da[plt_da['flag_clear'] == 'N']
    
    nb_historical_days = dir_path_recursive.split('/')[4]

    fig = plt.figure()
    gs = fig.add_gridspec(4, 15)
    
    ax00 = fig.add_subplot(gs[0:1, 0:3])    
    ax00.scatter(x=plt_da['ghi_obs'],y=plt_da['ghi_mod'],c='red',s=0.1,alpha=0.5)
    plt.plot(x_new,x_new,c='gray',linewidth=0.1)
    ax00.text( 50, 1000,'RMSE = %0.2f'%fitting(plt_da)[7], fontsize=3, color='red')
    ax00.text( 50, 1100,'R* = %0.2f'%fitting(plt_da)[0], fontsize=3, color='red')
    ax00.text( 200, 1390,'WRF-Solar '+ str(res) + '-km', fontsize=5, color='red')
    ax00.text(300, 1590, f'Year: {year} | training period: {nb_historical_days}',  fontsize=5)
    ax00.tick_params(axis='both', which='major', labelsize=5)
    ax00.set_ylim(0,1360)
    ax00.set_xlim(0,1360)
    ax00.set_ylabel('Model\nGHI (W/m$^2$)',  fontsize=5)
        
    
    ax00 = fig.add_subplot(gs[0:1, 8:11])
    ax00.scatter(x=plt_da['ghi_obs'],y=plt_da['GHI_pred_improved_fin'],c='c',s=0.1,alpha=0.5)
    plt.plot(x_new,x_new,c='gray',linewidth=0.1)
    ax00.tick_params(axis='both', which='major', labelsize=5)
    ax00.text( 50, 1100,'R* = %0.2f'%fitting_pred(plt_da)[0], fontsize=3, color='c')
    ax00.text( 50, 1000,'RMSE = %0.2f'%fitting_pred(plt_da)[7], fontsize=3, color='c')  
    ax00.text( 200, 1390,'KF+ WRF-Solar '+ str(res) + '-km', fontsize=5, color='c')
    ax00.set_ylim(0,1360)
    ax00.set_xlim(0,1360)
    ax00.text( -500, -800,'Observed\nGHI (W/m$^2$)', fontsize=5, color='k')

        
    ax00 = fig.add_subplot(gs[0:1, 4:7])
    ax00.scatter(x=plt_da['ghi_obs'],y=plt_da['GHI_pred_kf_only'],c='green',s=0.1,alpha=0.6)
    plt.plot(x_new,x_new,c='gray',linewidth=0.1)
    ax00.tick_params(axis='both', which='major', labelsize=5)
    ax00.text( 50, 1100,'R* = %0.2f'%kf_fitting_pred(plt_da)[0], fontsize=3, color='green')
    ax00.text( 50, 1000,'RMSE = %0.2f'%kf_fitting_pred(plt_da)[7], fontsize=3, color='green')  
    ax00.text( 200, 1390,'KF WRF-Solar '+ str(res) + '-km', fontsize=5, color='green')
    ax00.set_ylim(0,1360)
    ax00.set_xlim(0,1360)
    ax00.text( -800, 1600,'Cloudy Periods', fontsize=5, color='k')
    
    plt.savefig(f"{dir_path_recursive}/{year}_{nb_historical_days}_all_cloudy_kf_Manila_scatplot_" + str((res)) + "km.png", dpi=500, frameon=False, facecolor='white', bbox_inches="tight")

    with open(f'{dir_path_recursive}/{year}_{nb_historical_days}.csv',  'a') as f_object:
        writerObj = csv.writer(f_object)
        wrf = ('WRF cloudy all',fitting(plt_da)[0],fitting(plt_da)[7],fitting(plt_da)[8])
        kf = ('KF cloudy all',fitting_pred(plt_da)[0],fitting_pred(plt_da)[7],fitting_pred(plt_da)[8])
        writerObj.writerow(wrf)
        writerObj.writerow(kf)
        f_object.close()
        
############################
############################
############################ 
def scat_cloudy_cut(res, da_mean):

    x_new = np.linspace(0.01, 1360, 1000)
    plt_da = da_mean.iloc[nb_LTs*nb_historical_days:,:] #
    plt_da = plt_da[plt_da['flag_clear'] == 'N']
    
    fig = plt.figure()
    gs = fig.add_gridspec(4, 15)
    
    ax00 = fig.add_subplot(gs[0:1, 0:3])
    ax00.scatter(x=plt_da['ghi_obs'],y=plt_da['ghi_mod'],c='red',s=0.1,alpha=0.5)
    plt.plot(x_new,x_new,c='gray',linewidth=0.1)
    ax00.text( 50, 1000,'RMSE = %0.2f'%fitting(plt_da)[7], fontsize=3, color='red')
    ax00.text( 50, 1100,'R* = %0.2f'%fitting(plt_da)[0], fontsize=3, color='red')
    ax00.text( 200, 1390,'WRF-Solar '+ str(res) + '-km', fontsize=5, color='red')
    ax00.tick_params(axis='both', which='major', labelsize=5)
    ax00.set_ylim(0,1360)
    ax00.set_xlim(0,1360)
    ax00.set_ylabel('Model\nGHI (W/m$^2$)',  fontsize=5)
    
    ax00 = fig.add_subplot(gs[0:1, 8:11])
    ax00.scatter(x=plt_da['ghi_obs'],y=plt_da['GHI_pred_improved_fin'],c='c',s=0.1,alpha=0.5)
    plt.plot(x_new,x_new,c='gray',linewidth=0.1)
    ax00.tick_params(axis='both', which='major', labelsize=5)
    ax00.text( 50, 1100,'R* = %0.2f'%fitting_pred(plt_da)[0], fontsize=3, color='c')
    ax00.text( 50, 1000,'RMSE = %0.2f'%fitting_pred(plt_da)[7], fontsize=3, color='c')  
    ax00.text( 200, 1390,'KF+ WRF-Solar '+ str(res) + '-km', fontsize=5, color='c')
    ax00.set_ylim(0,1360)
    ax00.set_xlim(0,1360)
    ax00.text( -500, -800,'Observed\nGHI (W/m$^2$)', fontsize=5, color='k')

        
    ax00 = fig.add_subplot(gs[0:1, 4:7])
    ax00.scatter(x=plt_da['ghi_obs'],y=plt_da['GHI_pred_kf_only'],c='green',s=0.1,alpha=0.6)
    plt.plot(x_new,x_new,c='gray',linewidth=0.1)
    ax00.tick_params(axis='both', which='major', labelsize=5)
    ax00.text( 50, 1100,'R* = %0.2f'%kf_fitting_pred(plt_da)[0], fontsize=3, color='green')
    ax00.text( 50, 1000,'RMSE = %0.2f'%kf_fitting_pred(plt_da)[7], fontsize=3, color='green')  
    ax00.text( 200, 1390,'KF WRF-Solar '+ str(res) + '-km', fontsize=5, color='green')
    ax00.set_ylim(0,1360)
    ax00.set_xlim(0,1360)
    ax00.text( -800, 1600,'Cloudy Periods', fontsize=5, color='k')
    
    images_dir = 'img/2023_runs'
    plt.savefig(f"{images_dir}/{nb_historical_days}days/cut_cloudy_kf_Manila_scatplot_" + str((res)) + "km.png", dpi=500, frameon=False, facecolor='white', bbox_inches="tight")
    
    with open(f'{dir_path_recursive}/{year}_{nb_historical_days}.csv', 'a') as f_object:
        writerObj = csv.writer(f_object)
        wrf = ('WRF cloudy cut',fitting(plt_da)[0],fitting(plt_da)[7],fitting(plt_da)[8])
        kf = ('KF cloudy cut',fitting_pred(plt_da)[0],fitting_pred(plt_da)[7],fitting_pred(plt_da)[8])
        writerObj.writerow(wrf)
        writerObj.writerow(kf)
        f_object.close()
############################
############################
############################         
def scat_clear_all(res, da_mean,dir_path_recursive):

    x_new = np.linspace(0.01, 1360, 1000)
    plt_da = da_mean
    plt_da = plt_da[plt_da['flag_clear'] == 'Y']

    nb_historical_days = dir_path_recursive.split('/')[4]

    fig = plt.figure()
    gs = fig.add_gridspec(4, 15)
    
    ax00 = fig.add_subplot(gs[0:1, 0:3])

    ax00.text(300, 1590, f'Year: {year} | training period: {nb_historical_days}',  fontsize=5)

    ax00.scatter(x=plt_da['ghi_obs'],y=plt_da['ghi_mod'],c='red',s=0.1,alpha=0.5)
    plt.plot(x_new,x_new,c='gray',linewidth=0.1)
    ax00.text( 50, 1000,'RMSE = %0.2f'%fitting(plt_da)[7], fontsize=3, color='red')
    ax00.text( 50, 1100,'R* = %0.2f'%fitting(plt_da)[0], fontsize=3, color='red')
    ax00.text( 200, 1390,'WRF-Solar '+ str(res) + '-km', fontsize=5, color='red')
    ax00.tick_params(axis='both', which='major', labelsize=5)
    ax00.set_ylim(0,1360)
    ax00.set_xlim(0,1360)
    ax00.set_ylabel('Model\nGHI (W/m$^2$)',  fontsize=5) 
    
    ax00 = fig.add_subplot(gs[0:1, 8:11])
    ax00.scatter(x=plt_da['ghi_obs'],y=plt_da['GHI_pred_improved_fin'],c='c',s=0.1,alpha=0.5)
    plt.plot(x_new,x_new,c='gray',linewidth=0.1)
    ax00.tick_params(axis='both', which='major', labelsize=5)
    ax00.text( 50, 1100,'R* = %0.2f'%fitting_pred(plt_da)[0], fontsize=3, color='c')
    ax00.text( 50, 1000,'RMSE = %0.2f'%fitting_pred(plt_da)[7], fontsize=3, color='c')  
    ax00.text( 200, 1390,'KF+ WRF-Solar '+ str(res) + '-km', fontsize=5, color='c')
    ax00.set_ylim(0,1360)
    ax00.set_xlim(0,1360)
    ax00.text( -500, -800,'Observed\nGHI (W/m$^2$)', fontsize=5, color='k')

        
    ax00 = fig.add_subplot(gs[0:1, 4:7])
    ax00.scatter(x=plt_da['ghi_obs'],y=plt_da['GHI_pred_kf_only'],c='green',s=0.1,alpha=0.6)
    plt.plot(x_new,x_new,c='gray',linewidth=0.1)
    ax00.tick_params(axis='both', which='major', labelsize=5)
    ax00.text( 50, 1100,'R* = %0.2f'%kf_fitting_pred(plt_da)[0], fontsize=3, color='green')
    ax00.text( 50, 1000,'RMSE = %0.2f'%kf_fitting_pred(plt_da)[7], fontsize=3, color='green')  
    ax00.text( 200, 1390,'KF WRF-Solar '+ str(res) + '-km', fontsize=5, color='green')
    ax00.set_ylim(0,1360)
    ax00.set_xlim(0,1360)
    ax00.text( -800, 1600,'Clear Sky Periods', fontsize=5, color='k')
    
    plt.savefig(f"{dir_path_recursive}/{year}_{nb_historical_days}_all_clear_kf_Manila_scatplot_" + str((res)) + "km.png", dpi=500, frameon=False, facecolor='white', bbox_inches="tight")
   
    with open(f'{dir_path_recursive}/{year}_{nb_historical_days}.csv', 'a') as f_object:
        writerObj = csv.writer(f_object)
        wrf = ('WRF clear all',fitting(plt_da)[0],fitting(plt_da)[7],fitting(plt_da)[8])
        kf = ('KF clear all',fitting_pred(plt_da)[0],fitting_pred(plt_da)[7],fitting_pred(plt_da)[8])
        writerObj.writerow(wrf)
        writerObj.writerow(kf)
        f_object.close()
############################
############################
############################ 
def scat_clear_cut(res, da_mean):
    x_new = np.linspace(0.01, 1360, 1000)
    plt_da = da_mean.iloc[nb_LTs*nb_historical_days:,:] #
    plt_da = plt_da[plt_da['flag_clear'] == 'Y']
    fig = plt.figure()
    gs = fig.add_gridspec(4, 15)
    
    ax00 = fig.add_subplot(gs[0:1, 0:3])
    ax00.scatter(x=plt_da['ghi_obs'],y=plt_da['ghi_mod'],c='red',s=0.1,alpha=0.5)
    plt.plot(x_new,x_new,c='gray',linewidth=0.1)
    ax00.text( 50, 1000,'RMSE = %0.2f'%fitting(plt_da)[7], fontsize=3, color='red')
    ax00.text( 50, 1100,'R* = %0.2f'%fitting(plt_da)[0], fontsize=3, color='red')
    ax00.text( 200, 1390,'WRF-Solar '+ str(res) + '-km', fontsize=5, color='red')
    ax00.tick_params(axis='both', which='major', labelsize=5)
    ax00.set_ylim(0,1360)
    ax00.set_xlim(0,1360)
    ax00.set_ylabel('Model\nGHI (W/m$^2$)',  fontsize=5)
    
    
    ax00 = fig.add_subplot(gs[0:1, 8:11])
    ax00.scatter(x=plt_da['ghi_obs'],y=plt_da['GHI_pred_improved_fin'],c='c',s=0.1,alpha=0.5)
    plt.plot(x_new,x_new,c='gray',linewidth=0.1)
    ax00.tick_params(axis='both', which='major', labelsize=5)
    ax00.text( 50, 1100,'R* = %0.2f'%fitting_pred(plt_da)[0], fontsize=3, color='c')
    ax00.text( 50, 1000,'RMSE = %0.2f'%fitting_pred(plt_da)[7], fontsize=3, color='c')  
    ax00.text( 200, 1390,'KF+ WRF-Solar '+ str(res) + '-km', fontsize=5, color='c')
    ax00.set_ylim(0,1360)
    ax00.set_xlim(0,1360)
    ax00.text( -500, -800,'Observed\nGHI (W/m$^2$)', fontsize=5, color='k')

        
    ax00 = fig.add_subplot(gs[0:1, 4:7])
    ax00.scatter(x=plt_da['ghi_obs'],y=plt_da['GHI_pred_kf_only'],c='green',s=0.1,alpha=0.6)
    plt.plot(x_new,x_new,c='gray',linewidth=0.1)
    ax00.tick_params(axis='both', which='major', labelsize=5)
    ax00.text( 50, 1100,'R* = %0.2f'%kf_fitting_pred(plt_da)[0], fontsize=3, color='green')
    ax00.text( 50, 1000,'RMSE = %0.2f'%kf_fitting_pred(plt_da)[7], fontsize=3, color='green')  
    ax00.text( 200, 1390,'KF WRF-Solar '+ str(res) + '-km', fontsize=5, color='green')
    ax00.set_ylim(0,1360)
    ax00.set_xlim(0,1360)
    ax00.text( -800, 1600,'Clear Sky Periods', fontsize=5, color='k')    
    images_dir = 'img/2023_runs'
    plt.savefig(f"{images_dir}/{nb_historical_days}days/cut_clear_kf_Manila_scatplot_" + str((res)) + "km.png", dpi=500, frameon=False, facecolor='white', bbox_inches="tight")
   
    with open(f'{dir_path_recursive}/{year}_{nb_historical_days}.csv', 'a') as f_object:
        writerObj = csv.writer(f_object)
        wrf = ('WRF clear cut',fitting(plt_da)[0],fitting(plt_da)[7],fitting(plt_da)[8])
        kf = ('KF clear cut',fitting_pred(plt_da)[0],fitting_pred(plt_da)[7],fitting_pred(plt_da)[8])
        writerObj.writerow(wrf)
        writerObj.writerow(kf)
        f_object.close()




