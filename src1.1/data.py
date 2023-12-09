# Original authorauthor: Lu Li; 
# mail: lilu83@mail.sysu.edu.cn
# Modified by Qingliang Li, Cheng Zhang
# mail: liqingliang@ccsfu.edu.cn

from email import message
from locale import normalize
import os
import glob
import sys
import gc
import json
import datetime
import numpy as np
import xarray as xr
import h5py
import time
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

class Dataset():
    __name__ = ['fit', 'clip_by_date']
    def __init__(self, cfg: dict):
        """_summary_
        Args:
            s_resolution (str): spatial resolution of target  data;
            product (str): product name of target  data;
            selected_year (list):  data from begin year to end year
            test_year  (list): test data from begin year to end year        
            normalize (bool): if or not normalization
        """
        self.s_resolution = cfg['spatial_resolution']
        self.product = cfg['product']
        self.inputs_path = cfg['inputs_path']
        self.selected_year = cfg['selected_year']
        self.test_year = cfg['test_year']
        self.normalize = cfg['normalize']
        self.label = cfg['label']
        self.seq_len = cfg['seq_len']
        self.forcing_list = cfg['forcing_list']
        self.static_list = cfg['static_list']
        self.land_surface_list = cfg['land_surface_list']
        self.s_namedict = {
            '2m_temperature':'t2m', 
	    '10m_u_component_of_wind':'u10',  
	    '10m_v_component_of_wind':'v10',      
            'precipitation': 'tp',
	    'snow_depth_water_equivalent':'sd',
	    'surface_sensible_heat_flux':'sshf',
	    'soil_temperature_level_1':'stl1',
            'soil_temperature_level_2':'stl2',
            'soil_temperature_level_3':'stl3',
            'soil_temperature_level_4':'stl4',
	    'surface_pressure':'sp',
	    'specific_humidity':'Q',
            'surface_thermal_radiation_downwards_w_m2':'strd',
            'surface_solar_radiation_downwards_w_m2':'ssrd',
	    'total_runoff':'ro',
            'volumetric_soil_water_layer_1':'swvl1',
            'volumetric_soil_water_layer_2':'swvl2',
            'volumetric_soil_water_layer_3':'swvl3',
            'volumetric_soil_water_layer_4':'swvl4',
            'clay_0-5cm_mean':'Band1',
            'sand_0-5cm_mean':'Band1',
            'silt_0-5cm_mean':'Band1',
            'soil_water_capacity':'SC'}
    def fit(self,cfg):
        # Get the original data file path 
        PATH = self.inputs_path+self.product+'/'+str(self.s_resolution)+'/'
        data_path = cfg['nc_data_path']+ self.product+'/'+str(self.s_resolution)+'/'        
        print(PATH)
# ------------------------------------------------------------------------------------------------------------------------------
        print('[ATAI {d_p} work ] loading forcing'.format(d_p=cfg['workname']))

        begin_year = self.selected_year[0]
        end_year = self.selected_year[1]

	#Load forcing data
        forcing_list = []
        day_list = []

        for year in range(begin_year, end_year+1):
            file_name_forcing = 'ERA5-Land_forcing {sr} spatial resolution {year}.npy'.format(sr=self.s_resolution,year=year)
            if not os.path.exists(PATH+file_name_forcing):
                latitude, longitude = self._load_forcing_or_land_surface(data_path,
                                             self.forcing_list,
                                             self.s_resolution,
                                             self.s_namedict,
                                             year,
                                             cfg,
                                             PATH,
                                             file_name_forcing,
                                             category='atmosphere')
                lon = longitude
                lat = latitude
            data = np.load(PATH+file_name_forcing,mmap_mode='r')
            forcing_list.append(data)
            day_list.append(data.shape[0])
        lat_file_name = 'lat_{s}.npy'.format(
            s=self.s_resolution)
        lon_file_name = 'lon_{s}.npy'.format(
            s=self.s_resolution)
        if not os.path.exists(PATH+lat_file_name):
            np.save(PATH+lat_file_name, lat)
            np.save(PATH+lon_file_name, lon)
            
        else:
            lat = np.load(PATH+lat_file_name)
            lon = np.load(PATH+lon_file_name)      
	#Determine whether to perform memmap mapping   
	#Memmap is to deal with the problem of excessive amount of data             
        if cfg['memmap']:
            print("-------------------------")
            forcing = np.memmap(PATH+'forcing_memmap.npy',dtype=cfg['data_type'],mode='w+',shape=(np.sum(day_list,axis=0),forcing_list[0].shape[1],forcing_list[0].shape[2],forcing_list[0].shape[3]))
            start = 0
            end =0
            for i in range(len(forcing_list)):
                if i ==0:
                    start=0
                    end = day_list[i]
                else:
                    start = end
                    end = start+day_list[i]
                forcing[start:end] = forcing_list[i]
            forcing.flush()
            del forcing
        forcing = np.memmap(PATH+'forcing_memmap.npy',dtype=cfg['data_type'],mode='r',shape=(np.sum(day_list,axis=0),forcing_list[0].shape[1],forcing_list[0].shape[2],forcing_list[0].shape[3]))
	

        if os.path.exists(PATH+file_name_forcing+'/'+lat_file_name):
            lat, lon = np.load(PATH+lat_file_name), np.load(PATH+lon_file_name)
# ------------------------------------------------------------------------------------------------------------------------------
        print('[ATAI {d_p} work ] loading land surface data'.format(d_p=cfg['workname']))
	#Load land_surface data
        land_surface_list = []
        for year in range(begin_year, end_year+1):
            file_name_land_surface = 'ERA5-Land_land_surface {sr} spatial resolution {year}.npy'.format(sr=self.s_resolution,year=year)
            if not os.path.exists(PATH+file_name_land_surface):
                latitude, longitude = self._load_forcing_or_land_surface(data_path,
                                             self.land_surface_list,
                                             self.s_resolution,
                                             self.s_namedict,
                                             year,
                                             cfg,
                                             PATH,
                                             file_name_land_surface,
                                             category='land_surface')
            data = np.load(PATH+file_name_land_surface,mmap_mode='r')
            land_surface_list.append(data) 
        if cfg['memmap']:
            
            land_surface=np.memmap(PATH+'land_surface_memmap.npy',dtype=cfg['data_type'],
		mode='w+',shape=(np.sum(day_list,axis=0),land_surface_list[0].shape[1],
		land_surface_list[0].shape[2],land_surface_list[0].shape[3]))
            
            for i in range(len(land_surface_list)):
                if i ==0:
                    start=0
                    end = day_list[i]
                else:
                    start = end
                    end = start+day_list[i]
                land_surface[start:end] = land_surface_list[i]
            land_surface.flush()
            del land_surface

        land_surface = np.memmap(PATH+'land_surface_memmap.npy',dtype=cfg['data_type'],mode='r',shape=(np.sum(day_list,axis=0),land_surface_list[0].shape[1],land_surface_list[0].shape[2],land_surface_list[0].shape[3]))
	
# ------------------------------------------------------------------------------------------------------------------------------
        print('[ATAI {d_p} work ] loading label'.format(d_p=cfg['workname']))
	#Load label data 
	#The default label is volumetric_soil_water_layer_20
        label = []
        for year in range(begin_year, end_year + 1):
            file_name_label = 'ERA5_LAND_label_{sr}_{year}.npy'.format(sr=self.s_resolution,year=year)
            if not os.path.exists(PATH + file_name_label):
                lat, lon = self._load_forcing_or_land_surface(data_path,
                                                          self.label,
                                                          self.s_resolution,
                                                          self.s_namedict,
                                                          year,
                                                          cfg,
                                                          PATH,
                                                          file_name_label,
                                                          category='land_surface')
            data = np.load(PATH + file_name_label, mmap_mode='r')  # (t,lat,lon,feat)
            label.append(data)
        label = np.concatenate(label, axis=0)
        print('label type is {d_p}'.format(d_p=label.dtype))
        if cfg['data_type']=='float32':
            label = label.astype(np.float32)
            #filter Glacial region   
        print('label type is {d_p}'.format(d_p=label.dtype))
        file_name_mask = 'Mask with {sr} spatial resolution.npy'.format(sr=self.s_resolution)
        mask = np.ones(label[0,:,:,0].shape)
        mask[np.isnan(label[0,:,:,0])] = 0
        print("mask shape is ",mask.shape)
        np.save(PATH+file_name_mask, np.squeeze(mask))
        mask = np.load(PATH+file_name_mask)
# ------------------------------------------------------------------------------------------------------------------------------
        #static = np.ones((label.shape[1],label.shape[2],2))
        print(' ATAI {d_p} work ] loading ancillary'.format(d_p=cfg['workname']))
	#Load static_norm data
        if not os.path.exists(PATH + 'static_norm.npy'):
            static = []
            for i in range(len(self.static_list)):
                file_static = data_path + 'constants' + '/' + self.static_list[i] + '.nc'
                with xr.open_dataset(file_static) as f_:
                    static_data = f_[self.s_namedict[self.static_list[i]]]
                    static_data = self._lon_transform(static_data)
                    static_data = self._interp(static_data,mask)     
                static.append(static_data)
            static = np.stack(static, axis=-1)
            print("static shape is ",static.shape)
            if cfg['data_type']=='float32':
                static = static.astype(np.float32)
            static = self._spatial_normalize(static)
            print('static type is {d_p}'.format(d_p=static.dtype))
            np.save(PATH+'static_norm.npy', static)
        else:
            static = np.load(PATH+'static_norm.npy')

# ------------------------------------------------------------------------------------------------------------------------------
	#Partition and create datasets
        print('begin:{begin_year}, end:{end_year}'.format(
            begin_year=self.selected_year[0], end_year=self.selected_year[1]))
        print('forcing shape is {shape}'.format(shape=forcing.shape))
        print('land surface shape is {shape}'.format(shape=land_surface.shape))
        print('label shape is {shape}'.format(shape=label.shape))
        print('static shape is {shape}'.format(shape=static.shape))
        assert forcing.shape[0] == label.shape[0], "X(t) /= label(t)"
        # get shape
        self.time_length_f, self.nlat_f, self.nlon_f, self.num_features_f = forcing.shape  
        self.time_length_l, self.nlat_l, self.nlon_l, self.num_features_l = land_surface.shape
 
        #np.savetxt("/data/test/forcing.csv",forcing[-1,:,100,:],delimiter=",")
        #N = 365 * len(self.test_year)+self.seq_len
        N = 365 * len(self.test_year)+cfg['seq_len']+cfg['forcast_time']

        print('{n} samples for training, {m} samples for testing'.format(
            n=self.time_length_f-N, m=N))

        print('[ATAI {d_p} work ] preprocessing for generating train and test data'.format(d_p=cfg['workname']))
        print('\033[1;31m%s\033[0m' %
          'The following process are all time-consuming, especially for the high-resolution data. Please wait patiently')
	#Create a training dataset；The default training data is 1990 to 2019.
        if not os.path.exists(PATH+'x_train.npy'):
            x_train = np.memmap(PATH+'x_train.npy',dtype=cfg['data_type'],mode='w+',shape=(self.time_length_f-N, self.nlat_f, self.nlon_f, self.num_features_f+self.num_features_l))
            x_train[:,:,:,:self.num_features_f] = forcing[:self.time_length_f-N]
            x_train[:,:,:,self.num_features_f:] = land_surface[:self.time_length_f-N]
            x_train.flush()
            del x_train
        x_train = np.memmap(PATH+'x_train.npy',dtype=cfg['data_type'],mode='r+',shape=(self.time_length_f-N, self.nlat_f, self.nlon_f, self.num_features_f+self.num_features_l))
        print('[ATAI {d_p} work ] finish  generating x_train data, x_train shape is {xts}'.format(d_p=cfg['workname'],xts=x_train.shape))
# ------------------------------------------------------------------------------------------------------------------------------
        y_train = label[:self.time_length_f-N]
        np.save(PATH + 'y_train.npy', y_train)
        print('[ATAI {d_p} work ] finish  generating y_train data, y_train shape is {yts}'.format(d_p=cfg['workname'],yts=y_train.shape))
	#Create a test dataset ；The default for the test dataset is only 2020
        if not os.path.exists(PATH+'x_test.npy'):
            x_test = np.memmap(PATH+'x_test.npy',dtype=cfg['data_type'],mode='w+',shape=(N, self.nlat_f, self.nlon_f, self.num_features_f+self.num_features_l))
            x_test[:,:,:,:self.num_features_f],x_test[:,:,:,self.num_features_f:] = forcing[self.time_length_f-N:],land_surface[self.time_length_f-N:]
            x_test.flush()
            del x_test
        x_test = np.memmap(PATH+'x_test.npy',dtype=cfg['data_type'],mode='r+',shape=(N, self.nlat_f, self.nlon_f, self.num_features_f+self.num_features_l))
        print('[ATAI {d_p} work ] finish  generating x_test data, x_test shape is {xts}'.format(d_p=cfg['workname'],xts=x_test.shape))
        y_test = label[self.time_length_f-N:]
        print('[ATAI {d_p} work ] finish  generating y_test data, y_test shape is {yts}'.format(d_p=cfg['workname'],yts=y_train.shape))
        del forcing, label, land_surface
# ------------------------------------------------------------------------------------------------------------------------------
# normalize
        if self.normalize:
            if not os.path.exists(PATH+'x_train_norm.npy') or not os.path.exists(PATH+'y_train_norm.npy'): 
                print('ATAI {d_p} work ] start {nt} normalization forcing'.format(d_p=cfg['workname'],nt=cfg['normalize_type']))    
		#Adopt maximum and minimum normalization    
		#There are two forms of normalization：  region and gloabl                          
		#####region
                if cfg['normalize_type'] in ['region']:
                    scaler_x = np.memmap(PATH+'scaler_x.npy',dtype=cfg['data_type'],mode='w+',shape=(2, x_train.shape[1], x_train.shape[2], x_train.shape[3]))
                    scaler_y = np.memmap(PATH+'scaler_y.npy',dtype=cfg['data_type'],mode='w+',shape=(2, y_train.shape[1], y_train.shape[2], y_train.shape[3]))
                    for i in range (x_train.shape[2]):
                        out_x,out_y = self._get_minmax_scaler(x_train[:,:,i,:], y_train[:,:,i,:], scaler_x[:,:,i,:],scaler_y[:,:,i,:] ,'region')          
                        scaler_x[:,:,i,:],scaler_y[:,:,i,:] = out_x,out_y    
                    scaler_x.flush()
                    scaler_y.flush()
                    del scaler_x
                    del scaler_y
                    scaler_x = np.memmap(PATH+'scaler_x.npy',dtype=cfg['data_type'],mode='r+',shape=(2, x_train.shape[1], x_train.shape[2], x_train.shape[3]))
                    scaler_y = np.memmap(PATH+'scaler_y.npy',dtype=cfg['data_type'],mode='r+',shape=(2, y_train.shape[1], y_train.shape[2], y_train.shape[3]))
                    print('processed: x_train shape is: {x_s}, y_train shape is: {y_s}, x_test shape is: {x_ts_s}'.format(x_s=x_train.shape,y_s=y_train.shape,x_ts_s=x_test.shape))
                    
                    for i in range (x_train.shape[1]):
                        print('[{d_p}-th ] finish'.format(d_p=i))
                        out_x_train = self._normalize(x_train[:,i,:,:], 'input', scaler_x[:,i,:,:], 'minmax')
                        x_train[:,i,:,:] = out_x_train
                        out_y_train = self._normalize(y_train[:,i,:,:], 'output', scaler_y[:,i,:,:], 'minmax')
                        y_train[:,i,:,:] = out_y_train
                        out_x_test = self._normalize(x_test[:,i,:,:], 'input', scaler_x[:,i,:,:], 'minmax')
                        x_test[:,i,:,:] = out_x_test
                    
		####gloabl
                elif cfg['normalize_type'] in ['global']: 
                    scaler_x = np.memmap(PATH+'scaler_x.npy',dtype=cfg['data_type'],mode='w+',shape=(2, x_train.shape[3]))
                    scaler_y = np.memmap(PATH+'scaler_y.npy',dtype=cfg['data_type'],mode='w+',shape=(2, y_train.shape[3]))
                    scaler_y_t = {}
                    for i in range (x_train.shape[3]):
                        out_x,_ = self._get_minmax_scaler(x_train[:,:,:,i], x_train[:,:,:,i], scaler_x[:,i],scaler_y_t ,'global')
                        scaler_x[:,i] = np.squeeze(out_x)    
                    for i in range (y_train.shape[3]):
                        out_y,_ = self._get_minmax_scaler(y_train[:,:,:,i], y_train[:,:,:,i], scaler_y[:,i],scaler_y_t ,'global')                
                        scaler_y[:,i] = np.squeeze(out_y)      
                    scaler_x.flush()
                    scaler_y.flush()
                    del scaler_x
                    del scaler_y
                    scaler_x = np.memmap(PATH+'scaler_x.npy',dtype=cfg['data_type'],mode='r+',shape=(2,x_train.shape[3]))
                    scaler_y = np.memmap(PATH+'scaler_y.npy',dtype=cfg['data_type'],mode='r+',shape=(2,y_train.shape[3]))
                    print('processed: x_train shape is: {x_s}, y_train shape is: {y_s}, x_test shape is: {x_ts_s}'.format(x_s=x_train.shape,y_s=y_train.shape,x_ts_s=x_test.shape))
                    

                    for i in range (y_train.shape[3]):
                        print('[{d_p}-th ] for x_test finish'.format(d_p=i))
                        print('y_train shape is',y_train.shape)
                        scaler_y_in = np.expand_dims(scaler_y[:,i],axis=1)
                        scaler_y_in = np.expand_dims(scaler_y_in,axis=2)
                        print('scaler_y_in shape is',scaler_y_in.shape)
                        scaler_y_in = np.repeat(scaler_y_in,y_train.shape[1],axis=1)
                        scaler_y_in = np.repeat(scaler_y_in,y_train.shape[2],axis=2)
                        print('scaler_y_in shape is',scaler_y_in.shape)
                        out_y_train = self._normalize(y_train[:,:,:,i], 'output', scaler_y_in, 'minmax')
                        y_train[:,:,:,i] = out_y_train

                    for i in range (x_train.shape[3]):
                        print('[{d_p}-th ] for x_train finish'.format(d_p=i))
                        scaler_x_in = np.expand_dims(scaler_x[:,i],axis=1)
                        scaler_x_in = np.expand_dims(scaler_x_in,axis=2)
                        print('scaler_x_in shape is',scaler_x_in.shape)
                        scaler_x_in = np.repeat(scaler_x_in,x_train.shape[1],axis=1)
                        scaler_x_in = np.repeat(scaler_x_in,x_train.shape[2],axis=2)
                        print('scaler_x_in shape is',scaler_x_in.shape)
                        out_x_train = self._normalize(x_train[:,:,:,i], 'input', scaler_x_in, 'minmax')
                        x_train[:,:,:,i] = out_x_train
                        out_x_test = self._normalize(x_test[:,:,:,i], 'input', scaler_x_in, 'minmax')
                        x_test[:,:,:,i] = out_x_test


                    
# ------------------------------------------------------------------------------------------------------------------------------
# save
	    #Save the normalized dataset
            np.save(PATH+'x_train_norm_shape.npy', x_train.shape)
            np.save(PATH+'x_test_norm_shape.npy', x_test.shape)
            x_train_norm = np.memmap(PATH+'x_train_norm.npy',dtype=cfg['data_type'],mode='w+',shape=(x_train.shape))
            x_train_norm[:] = x_train[:]

            
            x_train_norm.flush()
            del x_train_norm          

            x_test_norm = np.memmap(PATH+'x_test_norm.npy',dtype=cfg['data_type'],mode='w+',shape=(x_test.shape))
            x_test_norm[:] = x_test[:]
            x_test_norm.flush()
            del x_test_norm

            y_train_norm = y_train            
            y_test_norm = y_test
            np.save(PATH + 'y_test_norm.npy', y_test_norm)            
            np.save(PATH + 'y_train_norm.npy', y_train_norm)
            x_train_norm = np.memmap(PATH+'x_train_norm.npy',dtype=cfg['data_type'],mode='r+',shape=(x_train.shape))

            
            x_test_norm = np.memmap(PATH+'x_test_norm.npy',dtype=cfg['data_type'],mode='r+',shape=(x_test.shape))


        return x_train_norm, y_train_norm, x_test_norm, y_test_norm, static, lat, lon, mask
#------------------------------------------------------------------------------------------------------------------------------
    #Convert nc data into npy data
    def _load_forcing_or_land_surface(self,
                      root,
                      _list,
                      s_resolution,
                      s_namedict,
                      year,
                      cfg,
                      PATH,
                      file_name,
                      category):

        tmp = []
        for i in range(len(_list)):
            file_ = root + category + '/' + str(year) + '/' + _list[i] + '.nc'
            with xr.open_dataset(file_) as f:
                    tmp.append(f[s_namedict[_list[i]]])
                    lat, lon = np.array(f.latitude), np.array(f.longitude)
        tmp = np.stack(tmp, axis=-1)
        np.save(PATH + file_name, tmp)
        print('ATAI {d_p} work ] finish loading {v_n} in {y} year '.format(d_p=cfg['workname'],v_n=_list, y=year))


        return lat, lon
#------------------------------------------------------------------------------------------------------------------------------
    def _normalize(self, feature, variable, scaler, scaler_type):
        if scaler_type == 'standard':
            if variable == 'input':
                feature = (
                    feature - np.array(scaler[0])) / np.array(scaler[1])
            elif variable == 'output':
                feature = (
                    feature - np.array(scaler[0])) / np.array(scaler[1])
            else:
                raise RuntimeError(f"Unknown variable type {variable}")
        elif scaler_type == 'minmax':
            if variable == 'input':
                feature = (feature - np.array(scaler[0])) / (
                    np.array(scaler[1])-np.array(scaler[0]))#?
            elif variable == 'output':
                feature = (feature - np.array(scaler[0])) / (
                    np.array(scaler[1])-np.array(scaler[0]))
            else:
                raise RuntimeError(f"Unknown variable type {variable}")
        return feature
#------------------------------------------------------------------------------------------------------------------------------
    #Reverse normalization, which is used in prediction
    def reverse_normalize(
            self,
            feature,
            variable: str,
            scaler,
            scaler_method: str,
            is_multivars: int) -> np.ndarray:
        """reverse normalized features using pre-computed statistics"""
        if variable == 'input':
            a, b = np.array(scaler[0]), np.array(scaler[1])
        elif variable == 'output':
            c, d = np.array(scaler[0]), np.array(scaler[1])
        if is_multivars != -1:
            a, b = a[:, :, is_multivars:is_multivars +
                     1], b[:, :, is_multivars:is_multivars+1]
            c, d = c[:, :, is_multivars:is_multivars +
                     1], d[:, :, is_multivars:is_multivars+1]
        if variable == 'input':
            if scaler_method == 'standard':
                feature = feature * b + a
            else:
                feature = feature * (b-a) + a
        elif variable == 'output':
            if scaler_method == 'standard':
                feature = feature * d + c#?
            else:
                feature = feature * (d-c) + c
        else:
            raise RuntimeError(f"Unknown variable type {variable}")
        return feature
#------------------------------------------------------------------------------------------------------------------------------
    def _get_minmax_scaler(self, X, y, scaler_x, scaler_y,type: str) -> dict:               
        if type == 'global':
            scaler_x[0] = np.squeeze(np.nanmin(
                X, axis=(0, 1, 2), keepdims=True).tolist())
            scaler_x[1] = np.squeeze(np.nanmax(
                X, axis=(0, 1, 2), keepdims=True).tolist())
            scaler_y = {}
        elif type == 'region':
            scaler_x[0] = np.nanmin(
                X, axis=(0), keepdims=True)  
            scaler_x[1] = np.nanmax(
                X, axis=(0), keepdims=True)
            scaler_y[0] = np.nanmin(
                y, axis=(0), keepdims=True)
            scaler_y[1] = np.nanmax(
                y, axis=(0), keepdims=True)
        else:
            raise IOError(f"Unknown variable type {type}")
        return scaler_x, scaler_y
#------------------------------------------------------------------------------------------------------------------------------
    def _spatial_normalize(self, static):
            # (ngrid, nfeat) for static data
            mean = np.nanmean(static, axis=(0,1), keepdims=True)
            std = np.nanstd(static, axis=(0,1), keepdims=True)
            return (static-mean)/std
#------------------------------------------------------------------------------------------------------------------------------
    def _lon_transform(self, x):
            x_new = np.zeros(x.shape)
            x_new[:,:int(x.shape[1]/2)] = x[:,int(x.shape[1]/2):] 
            x_new[:,int(x.shape[1]/2):] = x[:,:int(x.shape[1]/2)] 
            return x_new
#------------------------------------------------------------------------------------------------------------------------------
    def _interp(self, x, mask):
            x_ = np.ma.masked_invalid(x)
            arrange_lat = np.arange(0,x_.shape[0])
            arrange_lon = np.arange(0,x_.shape[1])
            lon_, lat_ = np.meshgrid(arrange_lon,arrange_lat)
            lat11_ = lat_[~x_.mask]
            lon11_ = lon_[~x_.mask]
            new_x = x_[~x_.mask].data
            inter_mean = np.nanmean(x_) 
            out = griddata((lon11_,lat11_),new_x.ravel(),(lon_,lat_),method='linear',fill_value=inter_mean)
            mask_value = x == x
            out[mask_value]=x[mask_value]
            out[mask==0]=np.nan 

            return out

