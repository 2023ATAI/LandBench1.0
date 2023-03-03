import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
from mpl_toolkits.basemap import Basemap
import os

import numpy as np
from config import get_args
# ---------------------------------# ---------------------------------
# configures

cfg = get_args()
out_path_lstm = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/' + cfg['workname'] + '/' + 'LSTM' +'/focast_time '+ str(cfg['forcast_time']) +'/'
out_path_CNN = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/' + cfg['workname'] + '/' + 'CNN' +'/focast_time '+ str(cfg['forcast_time']) +'/'
out_path_convlstm = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/' + cfg['workname'] + '/' + 'ConvLSTM' +'/focast_time '+ str(cfg['forcast_time']) +'/'

y_pred_convlstm = np.load(out_path_convlstm+'_predictions.npy')
name_convlstm = 'ConvLSTM'
y_pred_lstm = np.load(out_path_lstm+'_predictions.npy')
name_lstm = 'LSTM'
y_pred_cnn = np.load(out_path_CNN+'_predictions.npy')
name_cnn = 'CNN'
y_test = np.load(out_path_lstm+'observations.npy')
name_test = 'Observations'
pltday =  -121 # used for plt spatial distributions at 'pltday' day

r2_convlstm  = np.load(out_path_convlstm+'r2_'+'Convlstm'+'.npy')
r_convlstm  = np.load(out_path_convlstm+'r_'+'Convlstm'+'.npy')
urmse_convlstm  = np.load(out_path_convlstm+'urmse_'+'Convlstm'+'.npy')
r2_lstm  = np.load(out_path_lstm+'r2_'+'LSTM'+'.npy')
r_lstm  = np.load(out_path_lstm+'r_'+'LSTM'+'.npy')
urmse_lstm  = np.load(out_path_lstm+'urmse_'+'LSTM'+'.npy')
r2_cnn  = np.load(out_path_CNN+'r2_'+'CNN'+'.npy')
r_cnn  = np.load(out_path_CNN+'r_'+'CNN'+'.npy')
urmse_cnn  = np.load(out_path_CNN+'urmse_'+'CNN'+'.npy')



PATH = cfg['inputs_path']+cfg['product']+'/'+str(cfg['spatial_resolution'])+'/'
# gernate lon and lat
#lon_=np.linspace(73,135,y_pred_lstm.shape[2])
#np.save(PATH+'lon.npy',lon_)
#lat_=np.linspace(54,18,y_pred_lstm.shape[1])
#np.save(PATH+'lat.npy',lat_)

lon_ = np.load(PATH+'lon.npy')
lat_ = np.load(PATH+'lat.npy')
# Figure 5： configure for time series plot
sites_lon_index=[40,80,100,136]
sites_lat_index=[30,40,65,20]

# ---------------------------------
# Staitic 1： R2,ubrmse
# ---------------------------------
mask_data = r2_lstm[mask==1]
total_data = mask_data.shape[0]
#print('total_data  shape is', total_data.shape)
print('the average r2 of ConvLSTM model is :',np.nanmean(r2_convlstm[mask==1]))
print('the average r2 of LSTM model is :',np.nanmean(r2_lstm[mask==1]))
print('the average r2 of CNN model is :',np.nanmean(r2_cnn[mask==1]))

print('the average ubrmse of ConvLSTM model is :',np.nanmean(urmse_convlstm[mask==1]))
print('the average ubrmse of LSTM model is :',np.nanmean(urmse_lstm[mask==1]))
print('the average ubrmse of CNN model is :',np.nanmean(urmse_cnn[mask==1]))

print('the average r of ConvLSTM model is :',np.nanmean(r_convlstm[mask==1]))
print('the average r of LSTM model is :',np.nanmean(r_lstm[mask==1]))
print('the average r of CNN model is :',np.nanmean(r_cnn[mask==1]))

# ---------------------------------
# Figure 1： box plot
# ---------------------------------
# r2
# do mask
plt.figure
r2_convlstm_box = r2_convlstm[mask==1]
r2_lstm_box = r2_lstm[mask==1]
r2_cnn_box = r2_cnn[mask==1]
data_r2 = [r2_convlstm_box,r2_lstm_box,r2_cnn_box]

fig = plt.figure()
ax = plt.subplot(311)
plt.ylabel('R$^{2}$')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.boxplot(data_r2,
            notch=True,
            patch_artist=True,
            showfliers=False,
            labels=['ConvLSTM','LSTM','CNN'],
            boxprops=dict(facecolor='lightblue', color='black'))

# urmse
# do mask
urmse_convlstm_box = urmse_convlstm[mask==1]
urmse_lstm_box = urmse_lstm[mask==1]
urmse_cnn_box = urmse_cnn[mask==1]

data_urmse = [urmse_convlstm_box,urmse_lstm_box,urmse_cnn_box]

ax = plt.subplot(312)
plt.ylabel("urmse")
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.boxplot(data_urmse,
            notch=True,
            patch_artist=True,
            showfliers=False,
            labels=['ConvLSTM','LSTM','CNN'],
            boxprops=dict(facecolor='red', color='black'))

# r
# do mask
r_convlstm_box = r_convlstm[mask==1]
r_lstm_box = r_lstm[mask==1]
r_cnn_box = r_cnn[mask==1]

data_r = [r_convlstm_box,r_lstm_box,r_cnn_box]

ax = plt.subplot(313)
plt.ylabel("r")
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.boxplot(data_r,
            notch=True,
            patch_artist=True,
            showfliers=False,
            labels=['ConvLSTM','LSTM','CNN'],
            boxprops=dict(facecolor='green', color='black'))

plt.savefig(out_path_CNN+'box plot.png')
plt.show()
print('Figure 1 : box plot completed!')

# ------------------------------------------------------------------
# Figure 2： spatial distributions for predictions and observations
# ------------------------------------------------------------------

plt.figure
plt.subplot(2,2,1)



#global
#m = Basemap()
#m.drawcoastlines()
#m.drawcountries()
#parallels = np.arange(-90.,90,18.)
#meridians = np.arange(-180.,180.,36.)
#m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
#m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
#plt.show()

lon, lat = np.meshgrid(lon_, lat_)
m = Basemap(llcrnrlon=np.min(lon),
                llcrnrlat=np.min(lat),
                urcrnrlon=np.max(lon),
                urcrnrlat=np.max(lat))
#m.readshapefile("./china-shapefiles/china",  'china', drawbounds=False)
parallels = np.arange(0.,81,10.)
#m.drawparallels(parallels,labels=[False,True,True,False],dashes=[1,400])
meridians = np.arange(10.,351.,10.)
#m.drawmeridians(meridians,labels=[True,False,False,True],dashes=[1,400])
xi, yi = m(lon, lat)

# convlstm
plt.subplot(2,2,1)
y_pred_convlstm_pltday = y_pred_convlstm[pltday, :,:]
y_pred_convlstm_pltday[mask==0]=-9999
m.readshapefile("/home/liqingliang//ATAI/LandBench/src/china-shapefiles/china", 'china', drawbounds=True)
m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
cs = m.contourf(xi,yi, y_pred_convlstm_pltday, np.arange(0, 0.6, 0.05), cmap='YlGnBu')  
cbar = m.colorbar(cs, location='bottom', pad="10%")
cbar.set_label('m$^{3}$/m$^{3}$')
plt.title(name_convlstm)
#plt.savefig(out_path_LSTM + name_lstm + '_spatial distributions.png')
# lstm
plt.subplot(2,2,2)
y_pred_lstm_pltday = y_pred_lstm[pltday, :,:]
y_pred_lstm_pltday[mask==0]=-9999
m.readshapefile("/home/liqingliang//ATAI/LandBench/src/china-shapefiles/china", 'china', drawbounds=True)
m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
cs = m.contourf(xi,yi, y_pred_lstm_pltday, np.arange(0, 0.6, 0.05), cmap='YlGnBu')  
cbar = m.colorbar(cs, location='bottom', pad="10%")
cbar.set_label('m$^{3}$/m$^{3}$')
plt.title(name_lstm)
#plt.savefig(out_path_LSTM + name_lstm + '_spatial distributions.png')
# cnn
plt.subplot(2,2,3)
y_pred_cnn_pltday = y_pred_cnn[pltday, :,:]
y_pred_cnn_pltday[mask==0]=-9999

m.readshapefile("/home/liqingliang//ATAI/LandBench/src/china-shapefiles/china", 'china', drawbounds=True)
m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
cs = m.contourf(xi,yi, y_pred_cnn_pltday, np.arange(0, 0.6, 0.05), cmap='YlGnBu')  
cbar = m.colorbar(cs, location='bottom', pad="10%")
cbar.set_label('m$^{3}$/m$^{3}$')
plt.title(name_cnn)
#plt.savefig(out_path_CNN + name_cnn + '_spatial distributions.png')

# observations
plt.subplot(2,2,4)
y_test_pltday = y_test[pltday, :,:]
y_test_pltday[mask==0]=-9999
m.readshapefile("/home/liqingliang//ATAI/LandBench/src/china-shapefiles/china", 'china', drawbounds=True)
m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
cs = m.contourf(xi,yi, y_test_pltday, np.arange(0, 0.6, 0.05), cmap='YlGnBu')  
cbar = m.colorbar(cs, location='bottom', pad="10%")
cbar.set_label('m$^{3}$/m$^{3}$')
plt.title(name_test)
plt.savefig(out_path_CNN + name_test + '_spatial distributions.png')
print('Figure 2 : spatial distributions for predictions and observations completed!')
plt.show()
# ------------------------------------------------------------------
# Figure 3： spatial distributions for r2
# ------------------------------------------------------------------

plt.figure
lon, lat = np.meshgrid(lon_, lat_)
m = Basemap(llcrnrlon=np.min(lon),
                llcrnrlat=np.min(lat),
                urcrnrlon=np.max(lon),
                urcrnrlat=np.max(lat))
#m.readshapefile("./china-shapefiles/china",  'china', drawbounds=False)
parallels = np.arange(0.,81,10.)
#m.drawparallels(parallels,labels=[False,True,True,False],dashes=[1,400])
meridians = np.arange(10.,351.,10.)
#m.drawmeridians(meridians,labels=[True,False,False,True],dashes=[1,400])
xi, yi = m(lon, lat)

# convlstm
plt.subplot(2,2,1)
r2_convlstm[mask==0]=-9999
m.readshapefile("/home/liqingliang//ATAI/LandBench/src/china-shapefiles/china", 'china', drawbounds=True)
m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
cs = m.contourf(xi,yi, r2_convlstm, np.arange(-1, 1, 0.1), cmap='seismic')  
cbar = m.colorbar(cs, location='bottom', pad="10%")
cbar.set_label('R$^{2}$')
plt.title(name_convlstm)
plt.savefig(out_path_CNN + 'r2_'+ name_convlstm + '_spatial distributions.png')

# lstm
plt.subplot(2,2,2)
r2_lstm[mask==0]=-9999
m.readshapefile("/home/liqingliang//ATAI/LandBench/src/china-shapefiles/china", 'china', drawbounds=True)
m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
cs = m.contourf(xi,yi, r2_lstm, np.arange(-1, 1, 0.1), cmap='seismic')  
cbar = m.colorbar(cs, location='bottom', pad="10%")
cbar.set_label('R$^{2}$')
plt.title(name_lstm)
plt.savefig(out_path_CNN + 'r2_'+ name_lstm + '_spatial distributions.png')

# cnn
r2_cnn[mask==0]=-9999
plt.subplot(2,2,3)
m.readshapefile("/home/liqingliang//ATAI/LandBench/src/china-shapefiles/china", 'china', drawbounds=True)
m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
cs = m.contourf(xi,yi, r2_cnn, np.arange(-1, 1, 0.1), cmap='seismic')  
cbar = m.colorbar(cs, location='bottom', pad="10%")
cbar.set_label('R$^{2}$')
plt.title(name_cnn)
plt.savefig(out_path_CNN + 'r2_'+ name_cnn + '_spatial distributions.png')
print('Figure 3: spatial distributions for r2 completed!')
plt.show()

# ------------------------------------------------------------------
# Figure 4： spatial distributions for ubrmse
# ------------------------------------------------------------------

plt.figure
lon, lat = np.meshgrid(lon_, lat_)
m = Basemap(llcrnrlon=np.min(lon),
                llcrnrlat=np.min(lat),
                urcrnrlon=np.max(lon),
                urcrnrlat=np.max(lat))
#m.readshapefile("./china-shapefiles/china",  'china', drawbounds=False)
parallels = np.arange(0.,81,10.)
#m.drawparallels(parallels,labels=[False,True,True,False],dashes=[1,400])
meridians = np.arange(10.,351.,10.)
#m.drawmeridians(meridians,labels=[True,False,False,True],dashes=[1,400])
xi, yi = m(lon, lat)

# convlstm
plt.subplot(2,2,1)
urmse_convlstm[mask==0]=-9999
m.readshapefile("/home/liqingliang//ATAI/LandBench/src/china-shapefiles/china", 'china', drawbounds=True)
m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
cs = m.contourf(xi,yi, urmse_convlstm, np.arange(0, 0.2, 0.01), cmap='RdBu')  
cbar = m.colorbar(cs, location='bottom', pad="10%")
cbar.set_label('ubrmse(m$^{3}$/m$^{3}$)')
plt.title(name_convlstm)
plt.savefig(out_path_CNN + 'urmse_'+ name_convlstm + '_spatial distributions.png')

# lstm
plt.subplot(2,2,2)
urmse_lstm[mask==0]=-9999
m.readshapefile("/home/liqingliang//ATAI/LandBench/src/china-shapefiles/china", 'china', drawbounds=True)
m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
cs = m.contourf(xi,yi, urmse_lstm, np.arange(0, 0.2, 0.01), cmap='RdBu')  
cbar = m.colorbar(cs, location='bottom', pad="10%")
cbar.set_label('ubrmse(m$^{3}$/m$^{3}$)')
plt.title(name_lstm)
plt.savefig(out_path_CNN + 'urmse_'+ name_lstm + '_spatial distributions.png')

# cnn
urmse_cnn[mask==0]=-9999
plt.subplot(2,2,3)

m.readshapefile("/home/liqingliang//ATAI/LandBench/src/china-shapefiles/china", 'china', drawbounds=True)
m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
cs = m.contourf(xi,yi, urmse_cnn, np.arange(0, 0.2, 0.01), cmap='RdBu')  
cbar = m.colorbar(cs, location='bottom', pad="10%")
cbar.set_label('ubrmse(m$^{3}$/m$^{3}$)')
plt.title(name_cnn)
plt.savefig(out_path_CNN + 'urmse_'+ name_cnn + '_spatial distributions.png')
print('Figure 4: spatial distributions for ubrmse completed!')
plt.show()

# ------------------------------------------------------------------
# Figure 5： spatial distributions for r
# ------------------------------------------------------------------

plt.figure

lon, lat = np.meshgrid(lon_, lat_)
m = Basemap(llcrnrlon=np.min(lon),
                llcrnrlat=np.min(lat),
                urcrnrlon=np.max(lon),
                urcrnrlat=np.max(lat))
#m.readshapefile("./china-shapefiles/china",  'china', drawbounds=False)
parallels = np.arange(0.,81,10.)
#m.drawparallels(parallels,labels=[False,True,True,False],dashes=[1,400])
meridians = np.arange(10.,351.,10.)
#m.drawmeridians(meridians,labels=[True,False,False,True],dashes=[1,400])
xi, yi = m(lon, lat)

# convlstm
plt.subplot(2,2,1)
r_convlstm[mask==0]=-9999
m.readshapefile("/home/liqingliang//ATAI/LandBench/src/china-shapefiles/china", 'china', drawbounds=True)
m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
cs = m.contourf(xi,yi, r_convlstm, np.arange(0, 1, 0.05), cmap='seismic')  
cbar = m.colorbar(cs, location='bottom', pad="10%")
cbar.set_label('R$^{2}$')
plt.title(name_convlstm)
plt.savefig(out_path_CNN + 'r_'+ name_convlstm + '_spatial distributions.png')

# lstm
plt.subplot(2,2,2)
r_lstm[mask==0]=-9999
m.readshapefile("/home/liqingliang//ATAI/LandBench/src/china-shapefiles/china", 'china', drawbounds=True)
m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
cs = m.contourf(xi,yi, r_lstm, np.arange(0, 1, 0.05), cmap='seismic')  
cbar = m.colorbar(cs, location='bottom', pad="10%")
cbar.set_label('R$^{2}$')
plt.title(name_lstm)
plt.savefig(out_path_CNN + 'r_'+ name_lstm + '_spatial distributions.png')

# cnn
r_cnn[mask==0]=-9999
plt.subplot(2,2,3)

m.readshapefile("/home/liqingliang//ATAI/LandBench/src/china-shapefiles/china", 'china', drawbounds=True)
m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])
cs = m.contourf(xi,yi, r_cnn, np.arange(0, 1, 0.05), cmap='seismic')  
cbar = m.colorbar(cs, location='bottom', pad="10%")
cbar.set_label('R$^{2}$')
plt.title(name_cnn)
plt.savefig(out_path_CNN + 'r2_'+ name_cnn + '_spatial distributions.png')
print('Figure 5: spatial distributions for r completed!')
plt.show()

# ---------------------------------
# Figure 6： time series plot
# ---------------------------------
plt.figure
lon, lat = np.meshgrid(lon_, lat_)
m = Basemap(llcrnrlon=np.min(lon),
                llcrnrlat=np.min(lat),
                urcrnrlon=np.max(lon),
                urcrnrlat=np.max(lat))
#m.readshapefile("./china-shapefiles/china",  'china', drawbounds=False)
parallels = np.arange(0.,81,10.)
#m.drawparallels(parallels,labels=[False,True,True,False],dashes=[1,400])
meridians = np.arange(10.,351.,10.)
#m.drawmeridians(meridians,labels=[True,False,False,True],dashes=[1,400])
xi, yi = m(lon, lat)

m.readshapefile("/home/liqingliang//ATAI/LandBench/src/china-shapefiles/china", 'china', drawbounds=True)
m.drawparallels(parallels, labels=[False, True, True, False], dashes=[1, 400])
m.drawmeridians(meridians, labels=[True, False, False, True], dashes=[1, 400])

for lon_index,lat_index in zip(sites_lon_index,sites_lat_index):
    # ndarray
    lon=lon_[int(lon_index)]
    lat=lat_[int(lat_index)]
    plt.plot(lon, lat, marker='*', color='red', markersize=9)
plt.legend(loc=0)
plt.show()


data_all = [y_test,y_pred_convlstm,y_pred_lstm,y_pred_cnn]
color_list=['black','blue','red','yellow']
name_plt5 = ['observations','ConvLSTM','LSTM','CNN']
for lon_index,lat_index in zip(sites_lon_index,sites_lat_index):
    count=0
    fig, axs = plt.subplots(1,1,figsize=(15, 2))
    for data_f5plt in (data_all):        
        axs.plot(data_f5plt[:,lat_index,lon_index], color=color_list[count],label=name_plt5[count])
        axs.legend(loc=1)
        count = count+1
    axs.set_title('lat is {lat_v} and lon is {ln_v}'.format(lat_v=lat_[int(lat_index)],ln_v=lon_[int(lon_index)]))
print('Figure 6： time series plot completed!')
plt.show()

