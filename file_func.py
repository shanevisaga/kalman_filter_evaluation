from __const__ import *


def prep_input():
    obs_path = f'with_flags_{seas}.csv'
    a = pd.read_csv(obs_path)
    a['Time'] = pd.to_datetime(a['Time'])

    a['ghi_obs'] = a['SPN1_Total_Solar']
    a.columns
    
    mod_path = f'csv_{seas}/'
    all_files = glob.glob(os.path.join(mod_path, "*.csv"))

    mod_ = []

    for f in all_files:
        mod = pd.read_csv(f)
        mod['Time'] = pd.to_datetime(mod['time'])#.dt.tz_localize(tz)
        mod = mod.set_index('Time')
        t0 = mod.index[0] + datetime.timedelta(hours=21)
        tf = t0 + datetime.timedelta(hours=14)
        mod['plot'] = 'N'
        mod.loc[((mod.index >= t0) & (mod.index<= tf)), 'plot'] = 'Y'
        mod=mod[mod['plot']=='Y']
        mod = mod.reset_index()
        mod_.append(mod)
        
    mod_ = pd.concat(mod_)

    mod_ = mod_[['Time', 'ens', 'ghi', 'ghi_clear', 'swddni', 'coszen', 'swddif', 'temp',  'Q2_rel', 'u10', 'v10',  'station_name', 'domain']]
    mod_.columns = ['Time', 'ens', 'ghi_mod', 'ghi_clear', 'swddni', 'coszen', 'swddif', 'T2',  'Humi_rel', 'U10', 'V10', 'station_name', 'domain']

    # wind speed in both x/y direction
    if mod_['U10'].max() > 2:
        mod_['U10'] = mod_['U10'] / 10
        mod_['V10'] = mod_['V10'] / 10
    return(a,mod_)   


def prep_opt(mod_, a,ens,domain):
    
    mod_d01 = mod_[(mod_['ens']== ens) & (mod_['domain']== domain) & (mod_['station_name']== station_name)]
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
    df_temp =  df_temp[['Time', 'LT','CMP22_Total_Solar', 'SPN1_Total_Solar',\
           'SPN1_Diff_Solar', 'CGR4_IR', 'YY', 'MM', 'HH', 'mm',\
            'dhi', 'ghi_obs', 'sza', 'cossza', 'dni',  'GHI_in', 'DNI_in', 'DHI_in', 'cossza_b',\
           'SPN1_Total_Solar_N', 't2_lim', 'cossza_noon', 'FT_t', 'FT_TOA', 'FT_TOA_t', 't3_llim', 't3_ulim',\
           'Diffuse_Ratio', 'SPN1_Diff_Solar_N',\
           'sigma', 'ghi_cc_val', 'dhi_cc_val', 't1_lim', 'flag_clear',\
           'ens', 'domain','station_name', 'ghi_mod', 'ghi_clear',\
           'T2',  'Humi_rel', 'U10', 'V10', \
           'Error_rel', 'Kc_GHI_pred', 'Kc_GHI_obs', 'Kc_obs_bias',\
           'Kc_GHI_pred_improved']]
    df_temp
    return(df,df_temp)
    
    
    
