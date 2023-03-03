import numpy as np
import torch
from utils import r2_score
import time
from data_gen import erath_data_transform
import sys
from data import Dataset

def batcher_lstm(x_test, y_test, aux_test, seq_len,forcast_time):
    n_t, n_feat = x_test.shape
    n = (n_t-seq_len-forcast_time)
    x_new = np.zeros((n, seq_len, n_feat))*np.nan
    y_new = np.zeros((n,1))*np.nan
    aux_new = np.zeros((n,aux_test.shape[0]))*np.nan
    for i in range(n):
        x_new[i] = x_test[i:i+seq_len]
        y_new[i] = y_test[i+seq_len+forcast_time]
        aux_new[i] = aux_test
    return x_new, y_new, aux_new

def batcher_cnn(x_test, y_test, aux_test, seq_len,forcast_time,spatial_offset,i,j,lat_index,lon_index):
    x_test = x_test.transpose(0,3,1,2)
    y_test = y_test.transpose(0,3,1,2)
    aux_test = aux_test.transpose(2,0,1)
    n_t, n_feat, n_lat,n_lon = x_test.shape

    n = (n_t-seq_len-forcast_time)
    x_new = np.zeros((n, seq_len, n_feat,2*spatial_offset+1,2*spatial_offset+1))*np.nan
    y_new = np.zeros((n,1))*np.nan
    aux_new = np.zeros((n,aux_test.shape[0],2*spatial_offset+1,2*spatial_offset+1))*np.nan
    for ni in range(n):
        lat_index_bias = lat_index[i] + spatial_offset
        lon_index_bias = lon_index[j] + spatial_offset
        x_new[ni] = x_test[ni:ni+seq_len,:,lat_index[lat_index_bias-spatial_offset:lat_index_bias+spatial_offset+1],:][:,:,:,lon_index[lon_index_bias-spatial_offset:lon_index_bias+spatial_offset+1]]
        y_new[ni] = y_test[ni+seq_len+forcast_time,:,i,j]
        aux_new[ni] = aux_test[:,lat_index[lat_index_bias-spatial_offset:lat_index_bias+spatial_offset+1],:][:,:,lon_index[lon_index_bias-spatial_offset:lon_index_bias+spatial_offset+1]]
    return x_new, y_new, aux_new

def batcher_convlstm(x_test, y_test, aux_test, seq_len,forcast_time,spatial_offset,i,j,lat_index,lon_index):
    x_test = x_test.transpose(0,3,1,2)
    y_test = y_test.transpose(0,3,1,2)
    aux_test = aux_test.transpose(2,0,1)
    n_t, n_feat, n_lat,n_lon = x_test.shape

    n = (n_t-seq_len-forcast_time)
    x_new = np.zeros((n, seq_len, n_feat,2*spatial_offset+1,2*spatial_offset+1))*np.nan
    y_new = np.zeros((n,1))*np.nan
    aux_new = np.zeros((n,aux_test.shape[0],2*spatial_offset+1,2*spatial_offset+1))*np.nan
    for ni in range(n):
        lat_index_bias = lat_index[i] + spatial_offset
        lon_index_bias = lon_index[j] + spatial_offset
        x_new[ni] = x_test[ni:ni+seq_len,:,lat_index[lat_index_bias-spatial_offset:lat_index_bias+spatial_offset+1],:][:,:,:,lon_index[lon_index_bias-spatial_offset:lon_index_bias+spatial_offset+1]]
        y_new[ni] = y_test[ni+seq_len+forcast_time,:,i,j]
        aux_new[ni] = aux_test[:,lat_index[lat_index_bias-spatial_offset:lat_index_bias+spatial_offset+1],:][:,:,lon_index[lon_index_bias-spatial_offset:lon_index_bias+spatial_offset+1]]
    return x_new, y_new, aux_new


def test(x, y, static, scaler, cfg, model,device):
    cls = Dataset(cfg)          
    model.eval()
    if cfg['modelname'] in ['CNN', 'ConvLSTM']:
#	Splice x according to the sphere shape
        lat_index,lon_index = erath_data_transform(cfg, x)
        print('\033[1;31m%s\033[0m' % "Applied Model is {m_n}, we need to transform the data according to the sphere shape".format(m_n=cfg['modelname']))
    # get mean, std (1, ngrid, 6)
    #mean, std = np.array(scaler[0]), np.array(scaler[1])
    #mean = torch.from_numpy(mean).to(device)
    #std = torch.from_numpy(std).to(device)  
    if y.shape[0]-cfg["seq_len"]-cfg["forcast_time"]!=365:
        seq_len = 7
        x = x[5-1:]
        y_pred_ens = np.zeros((y.shape[0]-seq_len-cfg['forcast_time']+1-5, y.shape[1], y.shape[2]))*np.nan
        y_true = y[seq_len+cfg['forcast_time']+5-1:,:,:,0]
    else:

        y_pred_ens = np.zeros((y.shape[0]-cfg["seq_len"]-cfg['forcast_time'], y.shape[1], y.shape[2]))*np.nan
        y_true = y[cfg["seq_len"]+cfg['forcast_time']:,:,:,0]
    print('x shape is',x.shape)
    print('the true label shape is: {ts} and the predicton shape is: {ps}'.format(ts=y_true.shape, ps=y_pred_ens.shape))
    mask = y_true == y_true
    t_begin = time.time()
# ------------------------------------------------------------------------------------------------------------------------------              
    # for each grid by lstm model
    if cfg["modelname"] in ['LSTM']:
        count = 1
        for i in range(x.shape[1]):
            for j in range(x.shape[2]):
                x_new, y_new, static_new = batcher_lstm(x[:,i,j,:], y[:,i,j,:], static[i,j,:], cfg["seq_len"],cfg['forcast_time'])
                #if np.sum(x_new)!=0:
                    #print('x_new are',x_new)
                x_new = torch.from_numpy(x_new).to(device)
                static_new = torch.from_numpy(static_new).to(device)
                static_new = static_new.unsqueeze(1)
                static_new = static_new.repeat(1,x_new.shape[1],1)
                x_new = torch.cat([x_new, static_new], 2)
                pred = model(x_new, static_new)
                #
                #pred = pred*std[i, j, :]+mean[i, j, :] #(nsample,1,1)
                pred = pred.cpu().detach().numpy()
                pred = np.squeeze(pred)
               # print('pred  is',pred)
                if cfg["normalize"] and cfg['normalize_type'] in ['region']:
                    pred = cls.reverse_normalize(pred,'output',scaler[:,i,j,0],'minmax',-1)
                elif cfg["normalize"] and cfg['normalize_type'] in ['global']:
                    pred = cls.reverse_normalize(pred,'output',scaler,'minmax',-1)
              #  print('pred reverse is',pred)
                y_pred_ens[:,i,j]=pred
                if count % 1000 == 0:
                    print(time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime()))
                    print('\r',end="")                    
                    print('Remain {fs} thound predictions'.format(fs=(x.shape[1]*x.shape[2]-count)/1000))
                    sys.stdout.flush()
                time.sleep(0.0001)
                count = count+1
# ------------------------------------------------------------------------------------------------------------------------------              
    # for each grid by cnn model
    if cfg["modelname"] in ['CNN']:
        count = 1
        for i in range(x.shape[1]):
            for j in range(x.shape[2]):
                x_new, y_new, static_new = batcher_convlstm(x, y, static, cfg["seq_len"],cfg['forcast_time'],cfg["spatial_offset"],i,j,lat_index,lon_index)  
                x_new = np.nan_to_num(x_new)
                static_new = np.nan_to_num(static_new)
                x_new = torch.from_numpy(x_new).to(device)
                static_new = torch.from_numpy(static_new).to(device)
                # x_new = torch.cat([x_new, static_new], 1)
                x_new = x_new.squeeze(1)
                x_new = x_new.reshape(x_new.shape[0],x_new.shape[1]*x_new.shape[2],x_new.shape[3],x_new.shape[4])
                x_new = torch.cat([x_new, static_new], 1)
                pred = model(x_new, static_new)
                #
                #pred = pred*std[i, j, :]+mean[i, j, :] #(nsample,1,1)
                
                pred = pred.cpu().detach().numpy()
                pred = np.squeeze(pred)
                if cfg["normalize"] and cfg['normalize_type'] in ['region']:
                    pred = cls.reverse_normalize(pred,'output',scaler[:,i,j,0],'minmax',-1)
                elif cfg["normalize"] and cfg['normalize_type'] in ['global']:
                    pred = cls.reverse_normalize(pred,'output',scaler,'minmax',-1)
                y_pred_ens[:,i,j]=pred
                if count % 1000 == 0:
                    print(time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime()))
                    print('\r',end="")                    
                    print('Remain {fs} thound predictions'.format(fs=(x.shape[1]*x.shape[2]-count)/1000))
                    sys.stdout.flush()
                time.sleep(0.0001)
                count = count+1
# ------------------------------------------------------------------------------------------------------------------------------              
    # for each grid by convlstm model
    if cfg["modelname"] in ['ConvLSTM']:

        count = 1
        for i in range(x.shape[1]):
            for j in range(x.shape[2]):
                x_new, y_new, static_new = batcher_convlstm(x, y, static, cfg["seq_len"],cfg['forcast_time'],cfg["spatial_offset"],i,j,lat_index,lon_index)  
                x_new = np.nan_to_num(x_new)
                static_new = np.nan_to_num(static_new)
                x_new = torch.from_numpy(x_new).to(device)
                static_new = torch.from_numpy(static_new).to(device)
                static_new = static_new.unsqueeze(1)
                static_new = static_new.repeat(1,x_new.shape[1],1,1,1)
                # x_new = torch.cat([x_new, static_new], 1)
                pred = model(x_new, static_new,cfg)
                pred = pred.cpu().detach().numpy()
                pred = np.squeeze(pred)
                if cfg["normalize"] and cfg['normalize_type'] in ['region']:
                    pred = cls.reverse_normalize(pred,'output',scaler[:,i,j,0],'minmax',-1)
                elif cfg["normalize"] and cfg['normalize_type'] in ['global']:
                    pred = cls.reverse_normalize(pred,'output',scaler,'minmax',-1)
                y_pred_ens[:,i,j]=pred
                if count % 1000 == 0:
                    print(time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime()))
                    print('\r',end="")                    
                    print('Remain {fs} thound predictions'.format(fs=(x.shape[1]*x.shape[2]-count)/1000))
                    sys.stdout.flush()
                time.sleep(0.0001)
                count = count+1
# ----------------------------------------------------------------------------------------------------------------------------              
    t_end = time.time()
    print('y_pred_ens shape is',y_pred_ens.shape)
    print('scaler shape is',scaler.shape)
    y_true_mask = y_true[mask]
    y_pred_ens_mask = y_pred_ens[mask]
    print('y_true_mask shape is : {ts}'.format(ts=y_true_mask.shape))
    print('the true label shape is: {ts} and the predicton shape is: {ps}'.format(ts=y_true.shape, ps=y_pred_ens.shape))
    # log
    r2_ens = r2_score(y_true_mask, y_pred_ens_mask)
    R = np.zeros(y_true.shape[0])
    for i in range(y_true.shape[0]):
        obs = np.squeeze(y_true[i,:])
        pre = np.squeeze(y_pred_ens[i,:])
        R[i] = np.corrcoef(obs,pre)[0,1]
    print('\033[1;31m%s\033[0m' %
          "Median R2 {:.3f} time cost {:.2f}".format(np.nanmedian(r2_ens), t_end - t_begin))
    print('\033[1;31m%s\033[0m' %
          "Median R {:.3f} time cost {:.2f}".format(np.nanmedian(R), t_end - t_begin))
    return y_pred_ens,y_true


