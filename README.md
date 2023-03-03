Operation sequence

# 1.Download data to local

**The data is hosted [here](https://doi.org/10.11888/Atmos.tpdc.300294) with the following directory structure**
|----LandBench<br>
|----|----0.5<br>
|----|----|----atmosphere<br>
|----|----|----|----1979<br>
|----|----|----|----|----2m_temperature.nc<br>
|----|----|----|----|----10m_u_component_of_wind.nc<br>
|----|----|----|----|----10m_v_component_of_wind.nc<br>
|----|----|----|----|----precipitation.nc<br>
|----|----|----|----|----specific_humidity.nc<br>
|----|----|----|----|----surface_pressure.nc<br>
|----|----|----|----1980<br>
...<br>
...<br>
|----|----|----|---- 2020<br>
|----|----|----land_surface<br>
|----|----|----|----1979<br>
|----|----|----|----|----volumetric_soil_water_layer_1.nc<br>
|----|----|----|----|----volumetric_soil_water_layer_2.nc<br>
|----|----|----|----|----volumetric_soil_water_layer_3.nc<br>
|----|----|----|----|----volumetric_soil_water_layer_4.nc<br>
|----|----|----|----|----total_runoff.nc<br>
|----|----|----|----|----surface_theraml_radiation_downwards_w_m2.nc<br>
|----|----|----|----|----surface_solar_radiation_downwards_w_m2.nc<br>
|----|----|----|----|----surface_sensible_heat_flux.nc<br>
|----|----|----|----|----surface_latent_heat_flux.nc<br>
|----|----|----|----|----surface_net_theraml_radiation.nc<br>
|----|----|----|----|----surface_net_solar_radiation.nc<br>
|----|----|----|----|----soil_temperature_level_1.nc<br>
|----|----|----|----|----soil_temperature_level_2.nc<br>
|----|----|----|----|----soil_temperature_level_3.nc<br>
|----|----|----|----|----soil_temperature_level_4.nc<br>
|----|----|----|----|----snow_depth_water_equivalent.nc<br>
|----|----|----|----|----snow_cover.nc<br>
|----|----|----|----|----skin_temperature.nc<br>
|----|----|----|----|----leaf_area_index_low_vegetation.nc<br>
|----|----|----|----|----leaf_area_index_high_vegetation.nc<br>
|----|----|----|----|----forecast_albedo.nc<br>
|----|----|----|----1980<br>
...<br>
...<br>
|----|----|----|---- 2020<br>
|----|----|----constants<br>
|----|----|----|----clay_0-5cm_mean.nc<br>
|----|----|----|----clay_15-30cm_mean.nc<br>
|----|----|----|----clay_30-60cm_mean.nc<br>
|----|----|----|----clay_60-100cm_mean.nc<br>
|----|----|----|----clay_100-200cm_mean.nc<br>
|----|----|----|----sand_0-5cm_mean.nc<br>
|----|----|----|----sand_15-30cm_mean.nc<br>
|----|----|----|----sand_30-60cm_mean.nc<br>
|----|----|----|----sand_60-100cm_mean.nc<br>
|----|----|----|----sand_100-200cm_mean.nc<br>
|----|----|----|----silt_0-5cm_mean.nc<br>
|----|----|----|----silt_15-30cm_mean.nc<br>
|----|----|----|----silt_30-60cm_mean.nc<br>
|----|----|----|----silt_60-100cm_mean.nc<br>
|----|----|----|----silt_100-200cm_mean.nc<br>
|----|----|----|----landtype.nc<br>
|----|----|----|----soil_water_capacity.nc<br>
|----|----1(The directory structure is the same as 0.5)<br>
|----|----2(The directory structure is the same as 0.5)<br>
|----|----4(The directory structure is the same as 0.5)<br>

# 2.Modify relevant parameters in config.py, such as path

The relevant parameters in config.py are described below；

##### Path parameter

    parser.add_argument('--inputs_path', type=str, default='/data/test/')#Store training or test data files 存放训练或者测试数据文件
    parser.add_argument('--nc_data_path', type=str, default='/data/')#Store data files in nc format 存放nc格式的数据文件
    parser.add_argument('--product', type=str, default='LandBench')#Project name  LandBench 项目名称
    parser.add_argument('--workname', type=str, default='LandBench')
    parser.add_argument('--modelname', type=str, default='w_climatology')# Process;Persistence;w_climatology;LSTM;ConvLSTM;CNN
    parser.add_argument('--label',nargs='+', type=str, default=["volumetric_soil_water_layer_20"])#volumetric_soil_water_layer_1;surface_sensible_heat_flux;volumetric_soil_water_layer_20
    parser.add_argument('--stride', type=float, default=20)
    parser.add_argument('--data_type', type=str, default='float32')#The default data type is float32 默认的data类型是float32

  ##### data parameter

    	parser.add_argument('--selected_year', nargs='+', type=int, default=[1990,2020])
    	#Data selection 1990-2020 数据选取1990-2020
          # forcing SM:["2m_temperature","10m_u_component_of_wind","10m_v_component_of_wind","precipitation","surface_pressure","specific_humidity"]
          # forcing SSHF:["2m_temperature","10m_u_component_of_wind","10m_v_component_of_wind","precipitation","surface_pressure","specific_humidity"]
    parser.add_argument('--forcing_list', nargs='+', type=str, default=["2m_temperature","10m_u_component_of_wind","10m_v_component_of_wind","precipitation","surface_pressure","specific_humidity"])
          # land surface SM:["surface_solar_radiation_downwards_w_m2","surface_thermal_radiation_downwards_w_m2","soil_temperature_level_2"]
          # land surface SSHF:["surface_solar_radiation_downwards_w_m2","surface_thermal_radiation_downwards_w_m2"]
    parser.add_argument('--land_surface_list', nargs='+', type=str, default=["surface_solar_radiation_downwards_w_m2","surface_thermal_radiation_downwards_w_m2","soil_temperature_level_2"])
          # static SM: ["soil_water_capacity"]
          # static SSHF: ["soil_water_capacity"]
        parser.add_argument('--static_list', nargs='+', type=str, default=["soil_water_capacity"]) 
    
        parser.add_argument('--memmap', type=bool, default=True)#Determine whether to use memmap mapping 判断是否使用memmap映射
        parser.add_argument('--test_year', nargs='+', type=int, default=[2020])#The default test year is 2020.测试年份 默认是2020年
        parser.add_argument('--input_size', type=float, default=9)
        parser.add_argument('--spatial_resolution', type=float, default=1)
        #spatial_resolution We default to 1, but we can handle other forms of data, such as 0.5.
        parser.add_argument('--normalize', type=bool, default=True)
        parser.add_argument('--split_ratio', type=float, default=0.8)#The ratio of dividing the data 划分数据的比值
        parser.add_argument('--spatial_offset', type=float, default=3) #CNN	






    # model parameter
    	parser.add_argument('--normalize_type', type=str, default='region')
    	#normalize is divided into two ways: global and region.
        parser.add_argument('--forcast_time', type=float, default=1)
        parser.add_argument('--learning_rate', type=float, default=0.001)#The default learning rate is 0.001 默认学习率是0.001
        parser.add_argument('--hidden_size', type=float, default=128)
        parser.add_argument('--batch_size', type=float, default=64)
        parser.add_argument('--patience', type=int, default=10) 
        parser.add_argument('--seq_len', type=float, default=7) #CNN:1; ;LSTM:365 or 7;   
        parser.add_argument('--epochs', type=float, default=1000)#500
        parser.add_argument('--niter', type=float, default=300) #200
        parser.add_argument('--num_repeat', type=float, default=3)#How many models are trained? the default is three. 
        parser.add_argument('--dropout_rate', type=float, default=0.15)
        parser.add_argument('--input_size_cnn', type=float, default=57) #CNN (seq_len+1)*input_size
        parser.add_argument('--kernel_size', type=float, default=3) #CNN
        parser.add_argument('--stride_cnn', type=float, default=2) #CNN
        cfg = vars(parser.parse_args())




    # convert path to PosixPath object
    #cfg["forcing_root"] = Path(cfg["forcing_root"])
    #cfg["et_root"] = Path(cfg["et_root"])
    #cfg["attr_root"] = Path(cfg["attr_root"])

# 3.Run main.py to process data and start training

# 4.Run postprocess.py and eval.py for detailed analysis
