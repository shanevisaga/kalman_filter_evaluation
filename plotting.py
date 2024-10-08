from __const__ import *

def fitting(da):

    x_new = np.linspace(0.01, 1360, 1000)
    x_a=np.array(da['ghi_obs'])
    y_a=np.array(da['ghi_mod'])
    xaa = x_a[~np.isnan(x_a)]
    yaa = y_a[~np.isnan(y_a)]
    slope_a, intercept_a, r_value_a, p_value_a, std_err_a = stats.linregress(x_a,y_a)
    pearson_corr, pearson_pval = pearsonr(xaa,yaa)
    coef, p = kendalltau(xaa,yaa)
    rms = mean_squared_error(xaa, yaa, squared=False)
    mae = mean_absolute_error(xaa, yaa)
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
    rms = mean_squared_error(xaa, yaa, squared=False)
    mae = mean_absolute_error(xaa, yaa)  
    return (pearson_corr, pearson_pval, slope_a, intercept_a, r_value_a, p_value_a, std_err_a, rms, mae)
###########################

def kf_fitting_pred(da):

    x_new = np.linspace(0.01, 1360, 1000)
    x_a=np.array(da['ghi_obs'])
    y_a=np.array(da['GHI_pred_kf_only'])
    xaa = x_a[~np.isnan(x_a)]
    yaa = y_a[~np.isnan(y_a)]
    slope_a, intercept_a, r_value_a, p_value_a, std_err_a = stats.linregress(x_a,y_a)
    pearson_corr, pearson_pval = pearsonr(xaa,yaa)
    coef, p = kendalltau(xaa,yaa)
    rms = mean_squared_error(xaa, yaa, squared=False)
    mae = mean_absolute_error(xaa, yaa)
    return (pearson_corr, pearson_pval, slope_a, intercept_a, r_value_a, p_value_a, std_err_a, rms, mae)

def ts_all(res, da_mean,dir_path_recursive):
    da_mean = da_mean
    fig = plt.figure()
    gs = fig.add_gridspec(6, 3)

    ax00 = fig.add_subplot(gs[0:1, 0:5])
    ax00.plot(da_mean.index,da_mean['ghi_obs'],c='DarkBlue',linewidth=1.5)
    ax00.plot(da_mean.index,da_mean['ghi_mod'],c='red',linewidth=1.)
    ax00.tick_params(axis='both', which='major', labelsize=5)
    ax00.set_ylim(0,1360)

    nb_historical_days = dir_path_recursive.split('/')[4]
    ensemble_member = dir_path_recursive.split('/')[1]
    domain = dir_path_recursive.split('/')[2]


    ax00.text(100, 1600, f'Manila Observatory WRF-Solar validation( {res} km | initialized 00 UTC previous day)\ntraining period: {nb_historical_days}',  fontsize=5)
    #ax00.text(100, 1300, 'ensmean v obs',  fontsize=5)   
    ax00.text( 100, 1100,'WRF RMSE = %0.2f'%fitting(da_mean)[7], fontsize=5, color='red')
    ax00.legend(['Obs','WRF-' + str(res) + 'km'], bbox_to_anchor=(1.02, 0.98), prop={'size': 5},loc=2, borderaxespad=0.)
    ax00.set_xticklabels([])
    
    ax00 = fig.add_subplot(gs[2:3, 0:3])
    ax00.plot(da_mean.index,da_mean['ghi_obs'],c='DarkBlue',linewidth=1.)
    ax00.plot(da_mean.index,da_mean['GHI_pred_improved_fin'],c='c',linewidth=.5)
    ax00.tick_params(axis='both', which='major', labelsize=5)
    ax00.set_ylim(0,1360)
    ax00.text( 100, 1100,'KFv2-WRF RMSE = %0.2f'%fitting_pred(da_mean)[7], fontsize=5, color='c')
    ax00.legend(['Obs','KF+'], bbox_to_anchor=(1.02, 0.98), prop={'size': 5},loc=2, borderaxespad=0.)
    ax00.set_xticklabels([])
    
    ax00 = fig.add_subplot(gs[1:2, 0:3])
    ax00.plot(da_mean.index,da_mean['ghi_obs'],c='DarkBlue',linewidth=1.)
    ax00.plot(da_mean.index,da_mean['GHI_pred_kf_only'],c='green',linewidth=0.5)
    ax00.tick_params(axis='both', which='major', labelsize=5)
    ax00.set_ylim(0,1360)  
    ax00.text( 100, 1100,'KF-WRF RMSE = %0.2f'%kf_fitting_pred(da_mean)[7], fontsize=5, color='green')
    ax00.legend(['Obs','KF only'], bbox_to_anchor=(1.02, 0.98), prop={'size': 5},loc=2, borderaxespad=0.)
    ax00.set_xticklabels([])
    
    
    plt.savefig(f"{dir_path_recursive}/{ensemble_member}_{domain}_{nb_historical_days}_all_Manila_ts_" + str(res) +"km.png", dpi=500, facecolor='white', bbox_inches="tight")
    
    
    with open(f'{dir_path_recursive}/{ensemble_member}_{domain}_{nb_historical_days}.csv', 'w') as fileObj:
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
def ts_cut(res, da_mean,dir_path_recursive,nb_LTs):
    da_mean = da_mean.iloc[nb_LTs*nbhd:,:] 
    fig = plt.figure()
    gs = fig.add_gridspec(6, 3)

    ax00 = fig.add_subplot(gs[0:1, 0:5])
    ax00.plot(da_mean.index,da_mean['ghi_obs'],c='DarkBlue',linewidth=1.5)
    ax00.plot(da_mean.index,da_mean['ghi_mod'],c='red',linewidth=1.)
    ax00.tick_params(axis='both', which='major', labelsize=5)
    ax00.set_ylim(0,1360)

    nb_historical_days = dir_path_recursive.split('/')[4]
    ensemble_member = dir_path_recursive.split('/')[1]
    domain = dir_path_recursive.split('/')[2]


    ax00.text(100, 1600, f'Manila Observatory WRF-Solar validation( {res} km | initialized 00 UTC previous day)\ntraining period: {nb_historical_days}',  fontsize=5)
    #ax00.text(100, 1300, 'ensmean v obs',  fontsize=5)   
    ax00.text( 100, 1100,'WRF RMSE = %0.2f'%fitting(da_mean)[7], fontsize=5, color='red')
    ax00.legend(['Obs','WRF-' + str(res) + 'km'], bbox_to_anchor=(1.02, 0.98), prop={'size': 5},loc=2, borderaxespad=0.)
    ax00.set_xticklabels([])
    
    ax00 = fig.add_subplot(gs[2:3, 0:3])
    ax00.plot(da_mean.index,da_mean['ghi_obs'],c='DarkBlue',linewidth=1.)
    ax00.plot(da_mean.index,da_mean['GHI_pred_improved_fin'],c='c',linewidth=0.5)
    ax00.tick_params(axis='both', which='major', labelsize=5)
    ax00.set_ylim(0,1360)
    ax00.text( 100, 1100,'KFv2-WRF RMSE = %0.2f'%fitting_pred(da_mean)[7], fontsize=5, color='c')
    ax00.legend(['Obs','KF+'], bbox_to_anchor=(1.02, 0.98), prop={'size': 5},loc=2, borderaxespad=0.)
    ax00.set_xticklabels([])
    
    ax00 = fig.add_subplot(gs[1:2, 0:3])
    ax00.plot(da_mean.index,da_mean['ghi_obs'],c='DarkBlue',linewidth=1.)
    ax00.plot(da_mean.index,da_mean['GHI_pred_kf_only'],c='green',linewidth=0.5)
    ax00.tick_params(axis='both', which='major', labelsize=5)
    ax00.set_ylim(0,1360)  
    ax00.text( 100, 1100,'KF-WRF RMSE = %0.2f'%kf_fitting_pred(da_mean)[7], fontsize=5, color='green')
    ax00.legend(['Obs','KF only'], bbox_to_anchor=(1.02, 0.98), prop={'size': 5},loc=2, borderaxespad=0.)
    ax00.set_xticklabels([])
    
    
    plt.savefig(f"{dir_path_recursive}/{ensemble_member}_{domain}_{nb_historical_days}_all_Manila_ts_" + str(res) +"km_cut.png", dpi=500, facecolor='white', bbox_inches="tight")
    
    
    with open(f'{dir_path_recursive}/{ensemble_member}_{domain}_{nb_historical_days}_cut.csv', 'w') as fileObj:
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

def scat_cloudy_all(res, da_mean,dir_path_recursive):
    x_new = np.linspace(0.01, 1360, 1000)
    plt_da = da_mean
    plt_da = plt_da[plt_da['flag_clear'] == 'N']
    
    nb_historical_days = dir_path_recursive.split('/')[4]
    ensemble_member = dir_path_recursive.split('/')[1]
    domain = dir_path_recursive.split('/')[2]

    fig = plt.figure()
    gs = fig.add_gridspec(4, 15)
    
    ax00 = fig.add_subplot(gs[0:1, 0:3])    
    ax00.scatter(x=plt_da['ghi_obs'],y=plt_da['ghi_mod'],c='red',s=0.1,alpha=0.3)
    plt.plot(x_new,x_new,c='gray',linewidth=0.1)
    ax00.text( 50, 1000,'RMSE = %0.2f'%fitting(plt_da)[7], fontsize=3, color='k')
    ax00.text( 50, 1100,'R* = %0.2f'%fitting(plt_da)[0], fontsize=3, color='k')
    ax00.text( 200, 1390,'WRF-Solar '+ str(res) + '-km', fontsize=5, color='red')
    ax00.text(300, 1590, f'training period: {nb_historical_days}',  fontsize=5)
    ax00.tick_params(axis='both', which='major', labelsize=5)
    ax00.set_ylim(0,1360)
    ax00.set_xlim(0,1360)
    ax00.set_ylabel('Model\nGHI (W/m$^2$)',  fontsize=5)
        
    
    ax00 = fig.add_subplot(gs[0:1, 8:11])
    ax00.scatter(x=plt_da['ghi_obs'],y=plt_da['GHI_pred_improved_fin'],c='c',s=0.1,alpha=0.3)
    plt.plot(x_new,x_new,c='gray',linewidth=0.1)
    ax00.tick_params(axis='both', which='major', labelsize=5)
    ax00.text( 50, 1100,'R* = %0.2f'%fitting_pred(plt_da)[0], fontsize=3, color='k')
    ax00.text( 50, 1000,'RMSE = %0.2f'%fitting_pred(plt_da)[7], fontsize=3, color='k')  
    ax00.text( 200, 1390,'KF+ WRF-Solar '+ str(res) + '-km', fontsize=5, color='c')
    ax00.set_ylim(0,1360)
    ax00.set_xlim(0,1360)
    ax00.text( -500, -800,'Observed\nGHI (W/m$^2$)', fontsize=5, color='k')

        
    ax00 = fig.add_subplot(gs[0:1, 4:7])
    ax00.scatter(x=plt_da['ghi_obs'],y=plt_da['GHI_pred_kf_only'],c='green',s=0.1,alpha=0.3)
    plt.plot(x_new,x_new,c='gray',linewidth=0.1)
    ax00.tick_params(axis='both', which='major', labelsize=5)
    ax00.text( 50, 1100,'R* = %0.2f'%kf_fitting_pred(plt_da)[0], fontsize=3, color='k')
    ax00.text( 50, 1000,'RMSE = %0.2f'%kf_fitting_pred(plt_da)[7], fontsize=3, color='k')  
    ax00.text( 200, 1390,'KF WRF-Solar '+ str(res) + '-km', fontsize=5, color='green')
    ax00.set_ylim(0,1360)
    ax00.set_xlim(0,1360)
    ax00.text( 800, 1600,'Cloudy Periods', fontsize=5, color='k')
    
    plt.savefig(f"{dir_path_recursive}/{ensemble_member}_{domain}_{nb_historical_days}_all_cloudy_kf_Manila_scatplot_" + str((res)) + "km.png", dpi=500, facecolor='white', bbox_inches="tight")

    with open(f'{dir_path_recursive}/{ensemble_member}_{domain}_{nb_historical_days}.csv',  'a') as f_object:
        writerObj = csv.writer(f_object)
        wrf = ('WRF cloudy all',fitting(plt_da)[0],fitting(plt_da)[7],fitting(plt_da)[8])
        kf = ('KF cloudy all',fitting_pred(plt_da)[0],fitting_pred(plt_da)[7],fitting_pred(plt_da)[8])
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
    ensemble_member = dir_path_recursive.split('/')[1]
    domain = dir_path_recursive.split('/')[2]

    fig = plt.figure()
    gs = fig.add_gridspec(4, 15)
    
    ax00 = fig.add_subplot(gs[0:1, 0:3])

    ax00.text(300, 1590, f'training period: {nb_historical_days}',  fontsize=5)

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
    ax00.text( 800, 1600,'Clear Sky Periods', fontsize=5, color='k')
    
    plt.savefig(f"{dir_path_recursive}/{ensemble_member}_{domain}_{nb_historical_days}_all_clear_kf_Manila_scatplot_" + str((res)) + "km.png", dpi=500, facecolor='white', bbox_inches="tight")
   
    with open(f'{dir_path_recursive}/{ensemble_member}_{domain}_{nb_historical_days}.csv', 'a') as f_object:
        writerObj = csv.writer(f_object)
        wrf = ('WRF clear all',fitting(plt_da)[0],fitting(plt_da)[7],fitting(plt_da)[8])
        kf = ('KF clear all',fitting_pred(plt_da)[0],fitting_pred(plt_da)[7],fitting_pred(plt_da)[8])
        writerObj.writerow(wrf)
        writerObj.writerow(kf)
        f_object.close()
############################
############################
############################ 
