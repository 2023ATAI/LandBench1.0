import argparse
import pickle
from pathlib import PosixPath, Path
import time
import json
import os
import wandb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import xarray as xr
import netCDF4 as nc
from train import train
from eval import test
from data import Dataset
from config import get_args
import torch
# ------------------------------------------------------------------------------ 
# Original author : Qingliang Li, Cheng Zhang, 12/23/2022
# Edited by Jinlong Zhuu, Gan Li	
# Inspired by Lu Li
# ------------------------------------------------------------------------------

def main(cfg):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    device_ids = [0,1]
    print('Now we training {d_p} product in {sr} spatial resolution'.format(d_p=cfg['product'],sr=str(cfg['spatial_resolution'])))
    # ------------------------------------------------------------------------------------------------------------------------------
    # x_train: nt,nf,nlat,nlon; y_train:nt,nlat,nlon, static:nlat,nlon
    print('1 step:-----------------------------------------------------------------------------------------------------------------')
    print('[ATAI {d_p} work ] Make & load inputs'.format(d_p=cfg['workname']))
    path = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/'
    if not os.path.isdir (path):
        os.makedirs(path)
    #To determine whether the raw data in LandBench has been processed, it is necessary to convert it according to the requirements of the designed model.
    if os.path.exists(path+'/x_train_norm.npy'):
        print(' [ATAI {d_p} work ] loading input data'.format(d_p=cfg['workname']))
        x_train_shape = np.load(path+'x_train_norm_shape.npy',mmap_mode='r')
        x_train = np.memmap(path+'x_train_norm.npy',dtype=cfg['data_type'],mode='r+',shape=(x_train_shape[0],x_train_shape[1], x_train_shape[2], x_train_shape[3]))
        x_test_shape = np.load(path+'x_test_norm_shape.npy',mmap_mode='r')
        x_test = np.memmap(path+'x_test_norm.npy',dtype=cfg['data_type'],mode='r+',shape=(x_test_shape[0],x_test_shape[1], x_test_shape[2], x_test_shape[3]))
        y_train = np.load(path+'y_train_norm.npy',mmap_mode='r')
        y_test = np.load(path+'y_test_norm.npy',mmap_mode='r')
        static = np.load(path+'static_norm.npy')
        file_name_mask = 'Mask with {sr} spatial resolution.npy'.format(sr=cfg['spatial_resolution'])
        mask = np.load(path+file_name_mask)
        print('x_train[:,:,68,1] is',x_train[:,:,68,1])
        print('x_test[:,:,68,1] is',x_test[:,:,68,1])
        #np.savetxt("/data/test/x_trainxxx.csv",x_train[:,90,:,2],delimiter=",")

    else:     
        print('[ATAI {d_p} work ] making input data'.format(d_p=cfg['workname']))
        cls = Dataset(cfg) #FIXME: saving to input path
        x_train, y_train, x_test, y_test, static, lat, lon,mask = cls.fit(cfg)
    # load scaler for inverse
    if cfg['normalize_type'] in ['region']:                                      
        scaler_x = np.memmap(path+'scaler_x.npy',dtype=cfg['data_type'],mode='r+',shape=(2, x_train.shape[1], x_train.shape[2], x_train.shape[3]))
        scaler_y = np.memmap(path+'scaler_y.npy',dtype=cfg['data_type'],mode='r+',shape=(2, y_train.shape[1], y_train.shape[2], y_train.shape[3]))  
    elif cfg['normalize_type'] in ['global']:    
        scaler_x = np.memmap(path+'scaler_x.npy',dtype=cfg['data_type'],mode='r+',shape=(2, x_train.shape[3]))
        scaler_y = np.memmap(path+'scaler_y.npy',dtype=cfg['data_type'],mode='r+',shape=(2, y_train.shape[3]))  
    # ------------------------------------------------------------------------------------------------------------------------------
    print('2 step:-----------------------------------------------------------------------------------------------------------------')
    print('[ATAI {d_p} work ] Train & load {m_n} Model'.format(d_p=cfg['workname'],m_n=cfg['modelname']))
    print('[ATAI {d_p} work ] Wandb info'.format(d_p=cfg['workname']))
    #default = dict(  
    # model
    #model_name=cfg['modelname'],
    #hidden_size=cfg['hidden_size'],
    ## train
    #batch_size=cfg['batch_size'],
    #epochs=cfg['epochs'])
    #wandb.init(cfg['modelname'], config=default, allow_val_change=True)
# ------------------------------------------------------------------------------------------------------------------------------
    #for i in range (x_train_shape[3]):
    #    x_ = x_train[:,:,:,i]
    #    print(i,'-th x has', np.isnan(x_).sum()/1000,'thousands NAN values')
    #print('label has', np.isnan(y_train).sum()/1000,'thousands NAN values')
# ------------------------------------------------------------------------------------------------------------------------------
    #Model training
    out_path = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/' + cfg['workname'] + '/' + cfg['modelname'] +'/focast_time '+ str(cfg['forcast_time']) +'/'
    if not os.path.isdir (out_path):
        os.makedirs(out_path)   
    if os.path.exists(out_path+cfg['modelname'] +'_para.pkl'):
        print('[ATAI {d_p} work ] loading trained model'.format(d_p=cfg['workname'])) 
        model = torch.load(out_path+cfg['modelname']+'_para.pkl')
    else:
        # train 
        print('[ATAI {d_p} work ] training {m_n} model'.format(d_p=cfg['workname'],m_n=cfg['modelname'])) 
        for j in range(cfg["num_repeat"]):
            train(x_train, y_train, static, mask, scaler_x, scaler_y, cfg, j,path,out_path,device,device_ids)
            model = torch.load(out_path+cfg['modelname']+'_para.pkl')
        print('[ATAI {d_p} work ] finish training {m_n} model'.format(d_p=cfg['workname'],m_n=cfg['modelname']))   
    # ------------------------------------------------------------------------------------------------------------------------------
    print('3 step:-----------------------------------------------------------------------------------------------------------------')  
    print('[ATAI {d_p} work ] Make predictions by {m_n} Model'.format(d_p=cfg['workname'],m_n=cfg['modelname']))  
# ------------------------------------------------------------------------------------------------------------------------------
    print('x_test shape :',x_test.shape)
    print('y_test shape :',y_test.shape)
    print('static shape :',static.shape)    
    print('scaler_x shape is',scaler_x.shape)
    print('scaler_y shape is',scaler_y.shape)
    #Model testing
    y_pred, y_test = test(x_test, y_test, static, scaler_y, cfg, model,device) 
# ------------------------------------------------------------------------------------------------------------------------------   
# save predicted values and true values
    print('[ATAI {d_p} work ] Saving predictions by {m_n} Model and we hope to use "postprocess" and "evaluate" codes for detailed analyzing'.format(d_p=cfg['workname'],m_n=cfg['modelname']))
    np.save(out_path +'_predictions.npy', y_pred)
    np.save(out_path + 'observations.npy', y_test)


if __name__ == '__main__':
    cfg = get_args()
    main(cfg)
