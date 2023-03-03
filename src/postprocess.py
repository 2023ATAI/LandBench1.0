import os
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from utils import unbiased_rmse,_rmse,_bias
from config import get_args

def lon_transform(x):
  x_new = np.zeros(x.shape)
  x_new[:,:,:int(x.shape[2]/2)] = x[:,:,int(x.shape[2]/2):] 
  x_new[:,:,int(x.shape[2]/2):] = x[:,:,:int(x.shape[2]/2)] 
  return x_new




def postprocess(cfg):
    PATH = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/'
    file_name_mask = 'Mask with {sr} spatial resolution.npy'.format(sr=cfg['spatial_resolution'])
    mask = np.load(PATH+file_name_mask)
    if cfg['modelname'] in ['ConvLSTM']:
        out_path_convlstm = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/' + cfg['workname'] + '/' + cfg['modelname'] +'/focast_time '+ str(cfg['forcast_time']) +'/'
        y_pred_convlstm = np.load(out_path_convlstm+'_predictions.npy')
        y_test_convlstm = np.load(out_path_convlstm+'observations.npy')
        #y_pred_convlstm = y_pred_convlstm[cfg["seq_len"]:]
        #y_test_convlstm = y_test_convlstm[cfg["seq_len"]:]
        print(y_pred_convlstm.shape, y_test_convlstm.shape)
        # get shape
        nt, nlat, nlon = y_test_convlstm.shape    
        # mask
        #mask=y_test_lstm==y_test_convlstm
        # cal perf
        r2_convlstm = np.full(( nlat, nlon), np.nan)
        urmse_convlstm = np.full(( nlat, nlon), np.nan)
        r_convlstm = np.full(( nlat, nlon), np.nan)
        rmse_convlstm = np.full(( nlat, nlon), np.nan)
        bias_convlstm = np.full(( nlat, nlon), np.nan)
        for i in range(nlat):
            for j in range(nlon):
                if not (np.isnan(y_test_convlstm[:, i, j]).any()):
                    urmse_convlstm[i, j] = unbiased_rmse(y_test_convlstm[:, i, j], y_pred_convlstm[:, i, j])
                    #r2_convlstm[i, j] = r2_score(y_test_convlstm[:, i, j], y_pred_convlstm[:, i, j])
                    r_convlstm[i, j] = np.corrcoef(y_test_convlstm[:, i, j], y_pred_convlstm[:, i, j])[0,1]
                    rmse_convlstm[i, j] = _rmse(y_test_convlstm[:, i, j], y_pred_convlstm[:, i, j])
                    bias_convlstm[i, j] = _bias(y_test_convlstm[:, i, j], y_pred_convlstm[:, i, j])
        np.save(out_path_convlstm + 'r2_'+cfg['modelname']+'.npy', r2_convlstm)
        np.save(out_path_convlstm + 'r_'+cfg['modelname']+'.npy', r_convlstm)
        np.save(out_path_convlstm + 'rmse_'+cfg['modelname']+'.npy', rmse_convlstm)
        np.save(out_path_convlstm + 'bias_'+cfg['modelname']+'.npy', bias_convlstm)
        np.save(out_path_convlstm + 'urmse_'+cfg['modelname']+'.npy', urmse_convlstm)
        print('postprocess ove, please go on')
# ------------------------------------------------------------------------------------------------------------------------------
    if not cfg['label'] == ['volumetric_soil_water_layer_20'] and cfg['modelname'] in ['LSTM']:
        out_path_lstm = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/' + cfg['workname'] + '/' + cfg['modelname'] +'/focast_time '+ str(cfg['forcast_time']) +'/'
        y_pred_lstm = np.load(out_path_lstm+'_predictions.npy')
        y_test_lstm = np.load(out_path_lstm+'observations.npy')


        print(y_pred_lstm.shape, y_test_lstm.shape)
        # get shape
        nt, nlat, nlon = y_test_lstm.shape 
        # mask
        #mask=y_test_lstm==y_test_lstm
        # cal perf
        r2_lstm = np.full(( nlat, nlon), np.nan)
        urmse_lstm = np.full(( nlat, nlon), np.nan)
        r_lstm = np.full(( nlat, nlon), np.nan)
        rmse_lstm = np.full(( nlat, nlon), np.nan)
        bias_lstm = np.full(( nlat, nlon), np.nan)
        for i in range(nlat):
            for j in range(nlon):
                if not (np.isnan(y_test_lstm[:, i, j]).any()):
                    #print(' y_pred_lstm[:, i, j] is', y_pred_lstm[:, i, j])
                    #print(' y_test_lstm[:, i, j] is', y_test_lstm[:, i, j])
                    urmse_lstm[i, j] = unbiased_rmse(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                    #r2_lstm[i, j] = r2_score(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                    #print(' r2_lstm[i, j] is', r2_lstm[i, j])
                    r_lstm[i, j] = np.corrcoef(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])[0,1]
                    rmse_lstm[i, j] = _rmse(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
                    bias_lstm[i, j] = _bias(y_test_lstm[:, i, j], y_pred_lstm[:, i, j])
        np.save(out_path_lstm + 'r2_'+'LSTM'+'.npy', r2_lstm)
        np.save(out_path_lstm + 'r_'+'LSTM'+'.npy', r_lstm)
        np.save(out_path_lstm + 'rmse_'+cfg['modelname']+'.npy', rmse_lstm)
        np.save(out_path_lstm + 'bias_'+cfg['modelname']+'.npy', bias_lstm)
        np.save(out_path_lstm + 'urmse_'+'LSTM'+'.npy', urmse_lstm)
        print('postprocess ove, please go on')
# ------------------------------------------------------------------------------------------------------------------------------
    if cfg['modelname'] in ['CNN']:
        out_path_cnn = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/' + cfg['workname'] + '/' + cfg['modelname'] +'/focast_time '+ str(cfg['forcast_time']) +'/'
        y_pred_cnn = np.load(out_path_cnn+'_predictions.npy')
        y_test_cnn = np.load(out_path_cnn+'observations.npy')
        y_pred_cnn = y_pred_cnn[cfg["seq_len"]:]
        y_test_cnn = y_test_cnn[cfg["seq_len"]:]
        print(y_pred_cnn.shape, y_test_cnn.shape)
        # get shape
        nt, nlat, nlon = y_test_cnn.shape 
        # mask
        #mask=y_test_cnn==y_test_cnn
        # cal perf
        r2_cnn = np.full(( nlat, nlon), np.nan)    
        urmse_cnn = np.full(( nlat, nlon), np.nan)
        r_cnn = np.full(( nlat, nlon), np.nan)
        rmse_cnn = np.full(( nlat, nlon), np.nan)
        bias_cnn = np.full(( nlat, nlon), np.nan)
        for i in range(nlat):
            for j in range(nlon):
                if not (np.isnan(y_test_cnn[:, i, j]).any()):
                    urmse_cnn[i, j] = unbiased_rmse(y_test_cnn[:, i, j], y_pred_cnn[:, i, j])
                    #r2_cnn[i, j] = r2_score(y_test_cnn[:, i, j], y_pred_cnn[:, i, j])
                    r_cnn[i, j] = np.corrcoef(y_test_cnn[:, i, j], y_pred_cnn[:, i, j])[0,1]
                    rmse_cnn[i, j] = _rmse(y_test_cnn[:, i, j], y_pred_cnn[:, i, j])
                    bias_cnn[i, j] = _bias(y_test_cnn[:, i, j], y_pred_cnn[:, i, j])
        np.save(out_path_cnn + 'r2_'+'CNN'+'.npy', r2_cnn)
        np.save(out_path_cnn + 'r_'+'CNN'+'.npy', r_cnn)
        np.save(out_path_cnn + 'rmse_'+cfg['modelname']+'.npy', rmse_cnn)
        np.save(out_path_cnn + 'bias_'+cfg['modelname']+'.npy', bias_cnn)
        np.save(out_path_cnn + 'urmse_'+'CNN'+'.npy', urmse_cnn)
        print('postprocess ove, please go on')
# ------------------------------------------------------------------------------------------------------------------------------
    if cfg['modelname'] in ['Process']:
        print('start Process')
        out_path_cnn = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/' + cfg['workname'] + '/' + cfg['modelname'] +'/focast_time '+ str(cfg['forcast_time']) +'/'
        y_pred_cnn = np.load(out_path_cnn+'_predictions.npy')
        if cfg['label'] == ["volumetric_soil_water_layer_20"]:
                y_test_cnn_layer1 = np.load(out_path_cnn+'observations_layer1.npy')
                y_test_cnn_layer2 = np.load(out_path_cnn+'observations_layer2.npy')
                y_test_cnn = (y_test_cnn_layer1*7+13*y_test_cnn_layer2)/20
                np.save(out_path_cnn + 'observations.npy', y_test_cnn)
                y_pred_cnn = y_pred_cnn/1000
             
        else:	
                y_test = np.load(out_path_cnn+'observations.npy')
        if cfg['label'] == ["surface_sensible_heat_flux"]:
                y_pred_cnn = -(y_pred_cnn)/(86400*cfg['forcast_time'])
        y_pred_cnn = y_pred_cnn[1:]
        #y_pred_cnn = lon_transform(y_pred_cnn)
        y_test_cnn = np.load(out_path_cnn+'observations.npy')
        print(y_pred_cnn.shape, y_test_cnn.shape)
        # get shape
        nt, nlat, nlon = y_test_cnn.shape 
        # mask
        #mask=y_test_cnn==y_test_cnn
        # cal perf
        r2_cnn = np.full(( nlat, nlon), np.nan)    
        urmse_cnn = np.full(( nlat, nlon), np.nan)
        r_cnn = np.full(( nlat, nlon), np.nan)
        rmse_cnn = np.full(( nlat, nlon), np.nan)
        bias_cnn = np.full(( nlat, nlon), np.nan)
        for i in range(nlat):
            for j in range(nlon):
                if not (np.isnan(y_test_cnn[:, i, j]).any()):
                    urmse_cnn[i, j] = unbiased_rmse(y_test_cnn[:, i, j], y_pred_cnn[:, i, j])
                    #r2_cnn[i, j] = r2_score(y_test_cnn[:, i, j], y_pred_cnn[:, i, j])
                    r_cnn[i, j] = np.corrcoef(y_test_cnn[:, i, j], y_pred_cnn[:, i, j])[0,1]
                    rmse_cnn[i, j] = _rmse(y_test_cnn[:, i, j], y_pred_cnn[:, i, j])
                    bias_cnn[i, j] = _bias(y_test_cnn[:, i, j], y_pred_cnn[:, i, j])
        np.save(out_path_cnn + 'r2_'+'Process'+'.npy', r2_cnn)
        np.save(out_path_cnn + 'r_'+'Process'+'.npy', r_cnn)
        np.save(out_path_cnn + 'rmse_'+cfg['modelname']+'.npy', rmse_cnn)
        np.save(out_path_cnn + 'bias_'+cfg['modelname']+'.npy', bias_cnn)
        np.save(out_path_cnn + 'urmse_'+'Process'+'.npy', urmse_cnn)
        print('postprocess ove, please go on')
# ------------------------------------------------------------------------------------------------------------------------------
    if cfg['label'] == ['volumetric_soil_water_layer_20'] and not cfg['modelname'] in ['Process'] and not cfg['modelname'] in ['Persistence'] and not cfg['modelname'] in ['w_climatology']:
        print('LSTM ---> volumetric_soil_water_layer_20')
        out_path_cnn = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/' + cfg['workname'] + '/' + cfg['modelname'] +'/focast_time '+ str(cfg['forcast_time']) +'/'
        y_pred_cnn_layer1 = np.load(out_path_cnn+'_predictions_layer1.npy')
        y_pred_cnn_layer2 = np.load(out_path_cnn+'_predictions_layer2.npy')
        y_pred_cnn = (y_pred_cnn_layer1*7+13*y_pred_cnn_layer2)/20
        #y_pred_cnn = lon_transform(y_pred_cnn)
        y_test_cnn_layer1 = np.load(out_path_cnn+'observations_layer1.npy')
        y_test_cnn_layer2 = np.load(out_path_cnn+'observations_layer2.npy')
        y_test_cnn = (y_test_cnn_layer1*7+13*y_test_cnn_layer2)/20
        np.save(out_path_cnn + 'observations.npy', y_test_cnn)
        np.save(out_path_cnn + '_predictions.npy', y_pred_cnn)
        print(y_pred_cnn.shape, y_test_cnn.shape)
        # get shape
        nt, nlat, nlon = y_test_cnn.shape 
        # mask
        #mask=y_test_cnn==y_test_cnn
        # cal perf
        r2_cnn = np.full(( nlat, nlon), np.nan)    
        urmse_cnn = np.full(( nlat, nlon), np.nan)
        r_cnn = np.full(( nlat, nlon), np.nan)
        rmse_cnn = np.full(( nlat, nlon), np.nan)
        bias_cnn = np.full(( nlat, nlon), np.nan)
        for i in range(nlat):
            for j in range(nlon):
                if not (np.isnan(y_test_cnn[:, i, j]).any()):
                    urmse_cnn[i, j] = unbiased_rmse(y_test_cnn[:, i, j], y_pred_cnn[:, i, j])
                    #r2_cnn[i, j] = r2_score(y_test_cnn[:, i, j], y_pred_cnn[:, i, j])
                    r_cnn[i, j] = np.corrcoef(y_test_cnn[:, i, j], y_pred_cnn[:, i, j])[0,1]
                    rmse_cnn[i, j] = _rmse(y_test_cnn[:, i, j], y_pred_cnn[:, i, j])
                    bias_cnn[i, j] = _bias(y_test_cnn[:, i, j], y_pred_cnn[:, i, j])
        np.save(out_path_cnn + 'r2_'+'Process'+'.npy', r2_cnn)
        np.save(out_path_cnn + 'r_'+'Process'+'.npy', r_cnn)
        np.save(out_path_cnn + 'rmse_'+cfg['modelname']+'.npy', rmse_cnn)
        np.save(out_path_cnn + 'bias_'+cfg['modelname']+'.npy', bias_cnn)
        np.save(out_path_cnn + 'urmse_'+'Process'+'.npy', urmse_cnn)
        print('postprocess ove, please go on')
# ------------------------------------------------------------------------------------------------------------------------------
    if cfg['modelname'] in ['Persistence']:
        print('Persistence ---> volumetric_soil_water_layer_20')
        out_path = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/' + cfg['workname'] + '/' + cfg['modelname'] +'/focast_time '+ str(cfg['forcast_time']) +'/'
        if not os.path.isdir (out_path):
            os.makedirs(out_path)
        path = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/'
        if cfg['label'] == ['volumetric_soil_water_layer_20']:
            y_test_1 = np.load(path+ cfg['workname'] + '/' + 'y_test_norm_SM1.npy',mmap_mode='r')
            y_test_2 = np.load(path+ cfg['workname'] + '/' + 'y_test_norm_SM2.npy',mmap_mode='r')
            y_test = (y_test_1*7+13*y_test_2)/20
            np.save(out_path + 'observations.npy', y_test)
        else:
            y_test = np.load(path+'y_test_norm.npy',mmap_mode='r')
        print('y_test shape is',y_test.shape)
        y_test_cnn = y_test[cfg['seq_len']+cfg['forcast_time']:,:,:,0]
        np.save(out_path + 'observations.npy', y_test_cnn)
        print(y_test_cnn.shape)
        y_pred_cnn = y_test[cfg['seq_len']+cfg['forcast_time']-cfg['forcast_time']:y_test.shape[0]-cfg['forcast_time'],:,:,0]
        np.save(out_path + '_predictions.npy', y_pred_cnn)

        print(y_pred_cnn.shape, y_test_cnn.shape)
        # get shape
        nt, nlat, nlon = y_test_cnn.shape 
        # mask
        #mask=y_test_cnn==y_test_cnn
        # cal perf
        r2_cnn = np.full(( nlat, nlon), np.nan)    
        urmse_cnn = np.full(( nlat, nlon), np.nan)
        r_cnn = np.full(( nlat, nlon), np.nan)
        rmse_cnn = np.full(( nlat, nlon), np.nan)
        bias_cnn = np.full(( nlat, nlon), np.nan)
        for i in range(nlat):
            for j in range(nlon):
                if not (np.isnan(y_test_cnn[:, i, j]).any()):
                    urmse_cnn[i, j] = unbiased_rmse(y_test_cnn[:, i, j], y_pred_cnn[:, i, j])
                    #r2_cnn[i, j] = r2_score(y_test_cnn[:, i, j], y_pred_cnn[:, i, j])
                    r_cnn[i, j] = np.corrcoef(y_test_cnn[:, i, j], y_pred_cnn[:, i, j])[0,1]
                    rmse_cnn[i, j] = _rmse(y_test_cnn[:, i, j], y_pred_cnn[:, i, j])
                    bias_cnn[i, j] = _bias(y_test_cnn[:, i, j], y_pred_cnn[:, i, j])
        np.save(out_path + 'r2_'+'Persistence'+'.npy', r2_cnn)
        np.save(out_path + 'r_'+'Persistence'+'.npy', r_cnn)
        np.save(out_path + 'rmse_'+cfg['modelname']+'.npy', rmse_cnn)
        np.save(out_path + 'bias_'+cfg['modelname']+'.npy', bias_cnn)
        np.save(out_path + 'urmse_'+'Persistence'+'.npy', urmse_cnn)
        print('postprocess ove, please go on')
# ------------------------------------------------------------------------------------------------------------------------------
    if cfg['modelname'] in ['w_climatology']:
        out_path = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/' + cfg['workname'] + '/' + cfg['modelname'] +'/focast_time '+ str(cfg['forcast_time']) +'/'
        if not os.path.isdir (out_path):
            os.makedirs(out_path)
        path = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/'

        if cfg['label'] == ['volumetric_soil_water_layer_20']:
            y_pre_1 = np.load(path + cfg['workname'] + '/'+ 'y_train_SM1.npy')
            y_pre_2 = np.load(path + cfg['workname'] + '/'+ 'y_train_SM2.npy')
            y_pre = (y_pre_1*7+13*y_pre_2)/20
        else:
            y_pre = np.load(path + 'y_train.npy')

        if cfg['label'] == ['volumetric_soil_water_layer_20']:
            y_test_1 = np.load(path+ cfg['workname'] + '/' 'y_test_norm_SM1.npy',mmap_mode='r')
            y_test_2 = np.load(path+ cfg['workname'] + '/' 'y_test_norm_SM2.npy',mmap_mode='r')
            y_test = (y_test_1*7+13*y_test_2)/20
            np.save(out_path + 'observations.npy', y_test)
        else:
            y_test = np.load(path+'y_test_norm.npy',mmap_mode='r')

        y_test_cnn = y_test[cfg['seq_len']+cfg['forcast_time']:,:,:,0]
        print('y_test shape is',y_test_cnn.shape)
        np.save(out_path + 'observations.npy', y_test_cnn)
        y_pred_cnn = np.zeros((y_pre.shape))*np.nan
        print('y_pred_cnn shape is',y_pred_cnn.shape)
        data = y_pre
        num_years = data.shape[0]//365
        weekly_climat = np.zeros((num_years,52,data.shape[1],data.shape[2]))
        for year in range (num_years-1):
            year_data = data[year*365:(year+1)*365]
            weekly_climat_per_year = year_data[:-1,:,:].reshape((52,7,year_data.shape[1],year_data.shape[2]))
            weekly_climat[year] = np.nanmean(weekly_climat_per_year,axis=1)
           # print('weekly_climat[year] is',weekly_climat[year])
        weekly_mean =np.nanmean(weekly_climat,axis=0)
        weekly_results_ = np.repeat(weekly_mean,7,axis=0)   
        weekly_results = np.concatenate((weekly_results_,np.expand_dims(weekly_results_[-1,:,:],axis=0)),axis=0)
        np.save(out_path + '_predictions.npy', weekly_results)
        y_pred_cnn = weekly_results
        print(weekly_results.shape, y_test_cnn.shape)
        # get shape
        nt, nlat, nlon = y_test_cnn.shape 
        # mask
        #mask=y_test_cnn==y_test_cnn
        # cal perf
        r2_cnn = np.full(( nlat, nlon), np.nan)    
        urmse_cnn = np.full(( nlat, nlon), np.nan)
        r_cnn = np.full(( nlat, nlon), np.nan)
        rmse_cnn = np.full(( nlat, nlon), np.nan)
        bias_cnn = np.full(( nlat, nlon), np.nan)
        for i in range(nlat):
            for j in range(nlon):
                if not (np.isnan(y_test_cnn[:, i, j]).any()):
                    urmse_cnn[i, j] = unbiased_rmse(y_test_cnn[:, i, j], y_pred_cnn[:, i, j])
                    #r2_cnn[i, j] = r2_score(y_test_cnn[:, i, j], y_pred_cnn[:, i, j])
                    r_cnn[i, j] = np.corrcoef(y_test_cnn[:, i, j], y_pred_cnn[:, i, j])[0,1]
                    rmse_cnn[i, j] = _rmse(y_test_cnn[:, i, j], y_pred_cnn[:, i, j])
                    bias_cnn[i, j] = _bias(y_test_cnn[:, i, j], y_pred_cnn[:, i, j])
        np.save(out_path + 'r2_'+'w_climatology'+'.npy', r2_cnn)
        np.save(out_path + 'r_'+'w_climatology'+'.npy', r_cnn)
        np.save(out_path + 'rmse_'+cfg['modelname']+'.npy', rmse_cnn)
        np.save(out_path + 'bias_'+cfg['modelname']+'.npy', bias_cnn)
        np.save(out_path + 'urmse_'+'w_climatology'+'.npy', urmse_cnn)
        print('postprocess ove, please go on')
# ------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    cfg = get_args()
    postprocess(cfg)




               


