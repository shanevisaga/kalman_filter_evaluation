"""
This code makes diagnostic plots for raw and postprocessed WRF-Solar output
Postprocessing using the Kalman Filter is from Rafael Alvarenga's code (rafael.alvarenga@etu.univ-guyane.fr)
"""
import glob
import os
from file_func import *
from plotting import *
from __const__ import *

method='KF'
def kalmanfi(df, df_temp, nb_LTs):
    #for nb_historical_days in [3,7,14,21,28,35,42,49,56,63,70]:
    for nb_historical_days in [63]:
        
        timestep_enough_historical = (nb_historical_days*2)*nb_LTs  
        for t in tqdm(range(len(df_temp))):

            hour = df_temp.loc[t,'Time'].hour
            minutes = df_temp.loc[t,'Time'].minute

            if t >= timestep_enough_historical:
                # slice df_temp
                df_timestep = df_temp.loc[t-(nb_historical_days*nb_LTs)-(nb_historical_days*nb_LTs*recursive_calculation_covariance_matrices):t,:].copy()
            else:
                df_timestep = df_temp.loc[t:t+(nb_historical_days*nb_LTs)+(nb_historical_days*nb_LTs*recursive_calculation_covariance_matrices),:].copy()
                df_timestep = df_timestep[::-1].reset_index(drop=True)
            df_timestep = df_timestep.loc[(df_timestep['Time'].dt.hour == hour) & (df_timestep['Time'].dt.minute == minutes)]
            df_timestep = df_timestep.loc[~np.isnan(df_timestep['Kc_GHI_pred'])]
            df_timestep = df_timestep.loc[~np.isnan(df_timestep['Kc_obs_bias'])]

            if (len(df_timestep) == 0) & (len(df_timestep) <= nb_historical_days + (nb_historical_days*recursive_calculation_covariance_matrices) or (t not in df_timestep.index)):
                df_temp.loc[t,'Kc_GHI_pred_improved'] = np.nan  
                continue

            df_timestep = df_timestep.iloc[-(nb_historical_days+1)-(nb_historical_days*recursive_calculation_covariance_matrices):,:]
            df_timestep = df_timestep.reset_index(drop = True)

            # define prediction-bias variance matrix
            W = np.eye(len(predictors_vector))/1000

            # define measurement-bias variance matrix
            V = 0.01

            # define initial error covariance matrix
            Po = np.eye(len(predictors_vector))*5

            # define initial predicted bias
            xo = np.zeros(len(predictors_vector)).reshape(len(predictors_vector),1)

            measurement_GHI = []
            old_predicted_GHI = []
            improved_GHI = []
            ground_truths = []
            predicted_coefs = []

            for idx_i, i in enumerate(df_timestep.index):
                if recursive_calculation_covariance_matrices == True:
                    # --------------------------------------------
                    #  Calculate matrices of covariance of errors
                    # --------------------------------------------
                    if idx_i > nb_historical_days:
                        mean_w = sum(predicted_coefs[-(1+day)] - predicted_coefs[-(2+day)] for day in range(nb_historical_days))/nb_historical_days
                        mean_v = sum(measurement_GHI[-(1+day)] - improved_GHI[-(1+day)] for day in range(nb_historical_days))/nb_historical_days

                        # old method
                        W = np.diag(list((1/(nb_historical_days-1))*sum(((predicted_coefs[-(1+day)] - predicted_coefs[-(2+day)]) - mean_w)**2 for day in range(nb_historical_days)).reshape(len(predictors_vector),)))
                        V = (1/(nb_historical_days-1))*sum(((measurement_GHI[-(1+day)] - improved_GHI[-(1+day)]) - mean_v)**2 for day in range(nb_historical_days))

                        # # improved method (from Lynch, 2014 - Simplified method to derive the Kalman Filter covariance matrices to predict wind speeds from a NWP model)
                        # W = (1/(nb_historical_days-1))*sum(dot(((predicted_coefs[-(1+day)] - predicted_coefs[-(2+day)]) - mean_w),((predicted_coefs[-(1+day)] - predicted_coefs[-(2+day)]) - mean_w).T) for day in range(nb_historical_days))
                        # V = (1/(nb_historical_days-1))*sum(dot(((measurement_GHI[-(1+day)] - improved_GHI[-(1+day)]) - mean_v),((measurement_GHI[-(1+day)] - improved_GHI[-(1+day)]) - mean_v).T) for day in range(nb_historical_days))

                # ----------------------------
                #           Predict
                # ----------------------------

                if idx_i == 0:
                    # predicted mean bias
                    x_pred = np.zeros_like(xo)
                    if add_noise_in_predictions == True:
                        x_pred = xo + np.random.multivariate_normal(mean=[0.5]*len(predictors_vector), cov=W, size=1).reshape(-1,1)
                    else:
                        x_pred = xo

                    # predicted bias covariance matrix
                    P = Po + W

                else:
                    # predicted mean bias
                    if add_noise_in_predictions == True:
                        x_pred = x_pred + np.random.multivariate_normal(mean=[0]*len(predictors_vector), cov=W, size=1).reshape(-1,1)
                    else:
                        x_pred = x_pred
                    #x_pred[1:,0] = 0 # only bias is being tracked, our prediction model doesn't account for the other predictors

                    # predicted bias covariance matrix
                    P = P + W

                # ----------------------------
                #           Update
                # ----------------------------

                # compute transition matrix based on the current predictors for this timestep
                H = [df_timestep.loc[i,predictor] for predictor in predictors_vector[1:]]
                H = np.asarray([1] + H).reshape(1, len(predictors_vector))
                if nonlinear_predictions == True:
                    H = [predictor**idx_predictor for idx_predictor, predictor in enumerate(H)]

                predicted_coefs.append(x_pred)
                improved_GHI.append(dot(H,x_pred))
                df_timestep.loc[i,'Kc_GHI_pred_improved'] = improved_GHI[-1][0,0]

                # compute residual mean bias and residual bias covariance
                if add_noise_in_predictions == True:
                    new_measurement = df_timestep.loc[i,'Kc_GHI_obs'] + np.random.normal(loc = 0, scale = V)
                else:
                    new_measurement = df_timestep.loc[i,'Kc_GHI_obs']
                residual_mean = new_measurement - dot(H,x_pred)
                residual_covariance = dot(H, P).dot(H.T) + V

                # compute Kalman gain based on the transition matrix and residual covariance
                K = dot(P, H.T).dot(inv(residual_covariance)) # from documentation https://filterpy.readthedocs.io/en/latest/index.html#use
                K = np.nan_to_num(K, nan = 0)

                # update mean bias after incorporating measurements
                x_pred = x_pred + dot(K,residual_mean)

                # update bias covariance matrix after incorporating measurements
                #P = dot(K,H).dot(P) # from https://www.youtube.com/watch?v=W0gai93yhsM
                P = np.dot(np.eye(len(predictors_vector)) - dot(K,H),P)

                measurement_GHI.append(new_measurement)
                old_predicted_GHI.append(df_timestep.loc[i,'Kc_GHI_pred'])
                ground_truths.append(df_timestep.loc[i,'Kc_GHI_obs'])

            # assign final prediction
            df_temp.loc[t,'Kc_GHI_pred_improved'] = improved_GHI[-1][0,0]
        # calculate overall error metrics for this group of predictors
        df_temp['GHI_pred_improved'] = df_temp['Kc_GHI_pred_improved'] * df_temp['GHI_in']

        df_temp = df_temp.loc[~np.isnan(df_temp['GHI_pred_improved'])]
        df_temp = df_temp.loc[~np.isnan(df_temp['ghi_obs'])]

        rms = mean_squared_error(df_temp['ghi_obs'], df_temp['GHI_pred_improved'], squared=False)
        mae = mean_absolute_error(df_temp['ghi_obs'], df_temp['GHI_pred_improved'])
        mse = mean_squared_error(df_temp['ghi_obs'], df_temp['GHI_pred_improved'])
        mbe = np.mean(df_temp['GHI_pred_improved'] - df_temp['ghi_obs'])

        print('\n\n**************************')
        print(f'Predictors: {predictors_vector}')
        print(f'- RMS: {rms}')
        print(f'- MAE: {mae}')
        #print(f'- MSE: {mse}')
        print(f'- MBE: {mbe}')

        # assign post-processed timesteps to original dataframe
        copy = df_temp[['Time','GHI_pred_improved']]


        df_updated = df.merge(copy,how='outer',on="Time")
        df_updated['kf_obs'] = np.abs(df_updated['GHI_pred_improved'] - df_updated['ghi_obs'])
        df_updated['wrf_obs'] = np.abs(df_updated['ghi_mod'] - df_updated['ghi_obs'])

        #Kalman Filter Alone
        def kf_only(df):
            if ((df['GHI_pred_improved'] >= 0)):
                kf_only = df['GHI_pred_improved']
            else:
                kf_only = df['ghi_mod']
            return(kf_only)

        df_updated["GHI_pred_kf_only"] = df_updated.apply(kf_only, axis=1)

        #recording HITs and MISS for each LT or time of the day (Contingency Table)
        def lim(df):
            if ((df['GHI_pred_improved'] >= 0) & (df['kf_obs'] <= df['wrf_obs'])):
                lim = 'HIT' #by KF
            else:
                lim = 'MISS'
            return(lim)

        df_updated["flag_hit"] = df_updated.apply(lim, axis=1)

        table = df_updated.groupby(['LT','flag_hit']).agg({'flag_hit': ['count']}).droplevel(axis=1, level=0).reset_index()#.to_csv('summary.csv')
        table = pd.pivot_table(table, values='count', index=['LT'],columns=['flag_hit']).reset_index()#.to_csv('summary.csv')

        #this will be used to know whether the WRF or KF output is better for each LT or time of the day
        def hit(t):
            if ((t['HIT'] >= 0) & (t['HIT'] >= t['MISS'])):
                hit = 'KF'
            else:
                hit = 'WRF'
            return(hit)
        table["final"] = table.apply(hit, axis=1)

        df_updated = df_updated.merge(table,how='outer',on="LT")
        #depending on contingency table for each LT or time of the day 
        #use KF or WRF output
        def final_pred(df):
            if ((df['GHI_pred_improved'] >= 0) & (df['final'] == 'KF')):
                lim = df['GHI_pred_improved']
            else:
                lim = df['ghi_mod']
            return(lim)

        df_updated["GHI_pred_improved_fin"] = df_updated.apply(final_pred, axis=1)
        df_updated = df_updated.sort_values(by='Time').reset_index()
        #example for nb_historical_days=4
        #we are only correcting for 29 while 48 points are retained
        df_updated = df_updated[['Time',  'CMP22_Total_Solar', 'SPN1_Total_Solar',
               'SPN1_Diff_Solar', 'CGR4_IR', 'dhi', 'ghi_a', 'sza', 'cossza', 'dni',
               'MM', 'DD', 'HH', 'mm', 'GHI_in', 'DNI_in', 'DHI_in', 'cossza_b',
               'SPN1_Total_Solar_N', 't2_lim', 'cossza_noon', 'FT_t', 'FT_TOA',
               'FT_TOA_t', 't3_llim', 't3_ulim', 'Diffuse_Ratio', 'SPN1_Diff_Solar_N',
               'sigma', 'ghi_cc_val', 'dhi_cc_val', 't1_lim', 'flag_clear', 'ghi_obs',
               'ens', 'domain', 'station_name', 'ghi_mod', 'YY',
               'Error_rel', 'Kc_GHI_pred', 'Kc_GHI_obs', 'Kc_obs_bias', 'LT',
               'GHI_pred_kf_only', 'GHI_pred_improved_fin']]
        # calculate overall error metrics for this group of predictors
        df_updated = df_updated[df_updated['ghi_mod']  > 0]
        df_updated = df_updated.loc[~np.isnan(df_updated['GHI_pred_improved_fin'])]
        df_updated = df_updated.loc[~np.isnan(df_updated['ghi_obs'])]

        rms = mean_squared_error(df_updated['ghi_obs'], df_updated['GHI_pred_improved_fin'], squared=False)
        mae = mean_absolute_error(df_updated['ghi_obs'], df_updated['GHI_pred_improved_fin'])
        mse = mean_squared_error(df_updated['ghi_obs'], df_updated['GHI_pred_improved_fin'])
        mbe = np.mean(df_updated['GHI_pred_improved_fin'] - df_updated['ghi_obs'])

        dir_path_recursive = f'{main_dir_path_recursive}/{ens}/{domain}/{station_name}/{nb_historical_days}_day_{method}'
        os.makedirs(dir_path_recursive, exist_ok=True)

        nb_historical_days = dir_path_recursive.split('/')[4]
        ensemble_member = dir_path_recursive.split('/')[1]

        df_updated.to_csv(f"{dir_path_recursive}/{ensemble_member}_{domain}_{nb_historical_days}_df.csv")

        ts_all(res, df_updated,dir_path_recursive)
        scat_cloudy_all(res, df_updated,dir_path_recursive)
        scat_clear_all(res, df_updated,dir_path_recursive)

#for ens in ['ens0','ens1','ens2','ens3']:
for ens in ['ensmean']:
    for domain in ['d01']:
    #for domain in ['d01', 'd02']:
        
        if domain == 'd01':
            res = 5
        elif domain == 'd02':
            res = 1
        
        a = prep_input()[0]
        mod_ = prep_input()[1]

        os.makedirs(main_dir_path_recursive, exist_ok=True)

        df = prep_opt(mod_, a, ens, domain)[0]
        df_temp = prep_opt(mod_, a, ens, domain)[1]

        nb_LTs = len(np.unique(df_temp.LT))
        
        method='KF'
        
        kalmanfi(df, df_temp, nb_LTs)

