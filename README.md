# Overview

### LandBench 1.0: a benchmark dataset and evaluation metrics for data-driven land surface variables prediction
Qingliang Li, Cheng Zhang, Wei Shangguan, Zhongwang Wei, Hua Yuan, Jinlong Zhua, Xiaoning Li, Lu Li, Gan Li, Pingping Liu, Yongjiu Dai

Abstract-The advancements in deep learning methods have presented new opportunities and challenges for predicting land surface variables (LSVs) due to their similarity with computer sciences tasks. However, few researchers focus on the benchmark datasets for LSVs predictions that hampers fair comparisons of different data-driven deep learning models. Hence, we propose a LSVs benchmark dataset and prediction toolbox to boost research in data-driven LSVs modeling and improve the consistency of data-driven deep learning models for LSVs. LSVs benchmark dataset contains a large number of hydrology-related variables, such as global soil moisture, runoff, etc., which can verify the simulation of hydrological processes. Various global data from European Centre for Medium-Range Weather Forecasts reanalysis 5 (ERA5), ERA5-land, global gridded soil information (SoilGrid), soil moisture storage capacity (SMSC), and moderate-resolution imaging spectroradiometer (MODIS) datasets have been pre-processed into daily data at 0.5-, 1-, 2-, and 4-degree resolutions to facilitate their use in data-driven models. Simple statistical metrics, i.e., the root mean squared error and correlation coefficient, are chosen to evaluate the performance of different deep learning (DL) models, including convolutional neural network, long short-term memory and convolution long short-term memory models, with lead times of 1 and 5 days. A processed-based model serves as a physic baseline, soil moisture and surface sensible heat fluxes are taken as the target variables. The developed benchmark dataset and evaluation metrics for predicting LSVs using data-driven approaches, named as the LandBench toolbox, were implemented using Pytorch. This toolbox facilitates the reimplementation of existing methods, the development of novel predictive models, and the utilization of unified evaluation metrics. Additionally, the toolbox incorporates address mapping technology to enable high-resolution global predictions with constrained computing resources. We hope LandBench will not only serves as a standardized framework, fostering equitable model comparisons, but also provides indispensable data and a robust scientific foundation essential for advancing climate change research, disaster management, and sustainable development initiatives.

### Why use LandBench?

The advancements in deep learning methods have presented new opportunities and challenges for predicting **land surface variables (LSVs)** due to their similarity with computer sciences tasks.

However, the lack of a benchmark dataset hampers fair comparisons of different data-driven deep learning models for LSVs predictions. 

Hence, we propose a LSVs benchmark dataset and prediction toolbox to boost research in data-driven LSVs modeling and improve the consistency of data-driven deep learning models for LSVs.

**The LSVs benchmark dataset is hosted here(https://doi.org/10.11888/Atmos.tpdc.300294)**<br>
**The prediction toolbox is hosted here(https://github.com/2023ATAI/LandBench1.0)**

### Installation

LandBench works in [Python3.9.13](https://www.python.org/downloads/)<br>
In order to use the LandBench successfully, the following site-packages are required:

- pytorch 1.13.1
- pandas 1.4.4
- numpy 1.22.0
- scikit-learn 1.0.2
- scipy 1.7.3
- matplotlib 3.5.2
- xarray 2023.1.0
- netCDF4 1.6.2


The latest LandBench1.0 can work in 

linux-Ubuntu 18.04.6

### Prepare Data

**The data is hosted here(https://doi.org/10.11888/Atmos.tpdc.300294) with the following directory structure**<br>

```
|----LandBench
|----|----0.5
|----|----|----atmosphere
|----|----|----|----1979
|----|----|----|----|----2m_temperature.nc
|----|----|----|----|----10m_u_component_of_wind.nc
|----|----|----|----|----10m_v_component_of_wind.nc
|----|----|----|----|----precipitation.nc
|----|----|----|----|----specific_humidity.nc
|----|----|----|----|----surface_pressure.nc
|----|----|----|----1980
...
...
|----|----|----|---- 2020
|----|----|----land_surface
|----|----|----|----1979
|----|----|----|----|----volumetric_soil_water_layer_1.nc
|----|----|----|----|----volumetric_soil_water_layer_2.nc
|----|----|----|----|----volumetric_soil_water_layer_3.nc
|----|----|----|----|----volumetric_soil_water_layer_4.nc
|----|----|----|----|----total_runoff.nc
|----|----|----|----|----surface_theraml_radiation_downwards_w_m2.nc
|----|----|----|----|----surface_solar_radiation_downwards_w_m2.nc
|----|----|----|----|----surface_sensible_heat_flux.nc
|----|----|----|----|----surface_latent_heat_flux.nc
|----|----|----|----|----surface_net_theraml_radiation.nc
|----|----|----|----|----surface_net_solar_radiation.nc
|----|----|----|----|----soil_temperature_level_1.nc
|----|----|----|----|----soil_temperature_level_2.nc
|----|----|----|----|----soil_temperature_level_3.nc
|----|----|----|----|----soil_temperature_level_4.nc
|----|----|----|----|----snow_depth_water_equivalent.nc
|----|----|----|----|----snow_cover.nc
|----|----|----|----|----skin_temperature.nc
|----|----|----|----|----leaf_area_index_low_vegetation.nc
|----|----|----|----|----leaf_area_index_high_vegetation.nc
|----|----|----|----|----forecast_albedo.nc
|----|----|----|----1980
...
...
|----|----|----|---- 2020
|----|----|----constants
|----|----|----|----clay_0-5cm_mean.nc
|----|----|----|----clay_15-30cm_mean.nc
|----|----|----|----clay_30-60cm_mean.nc
|----|----|----|----clay_60-100cm_mean.nc
|----|----|----|----clay_100-200cm_mean.nc
|----|----|----|----sand_0-5cm_mean.nc
|----|----|----|----sand_15-30cm_mean.nc
|----|----|----|----sand_30-60cm_mean.nc
|----|----|----|----sand_60-100cm_mean.nc
|----|----|----|----sand_100-200cm_mean.nc
|----|----|----|----silt_0-5cm_mean.nc
|----|----|----|----silt_15-30cm_mean.nc
|----|----|----|----silt_30-60cm_mean.nc
|----|----|----|----silt_60-100cm_mean.nc
|----|----|----|----silt_100-200cm_mean.nc
|----|----|----|----landtype.nc
|----|----|----|----soil_water_capacity.nc
|----|----1(The directory structure is the same as 0.5)
|----|----2(The directory structure is the same as 0.5)
|----|----4(The directory structure is the same as 0.5)
```

### Prepare Config File

Usually, we use the config file in model training, testing and detailed analyzing.

The config file contains all necessary information, such as path,data,model, etc.

The config file of our work is `LandBench1.0/src/config.py`

### Process data and train model

Run the following command in the directory of `LandBench1.0/src/` to process data and start training.

```
python main.py 
```

### Detailed analyzing

Run the following command in the directory of `LandBench1.0/src/` to get detailed analyzing.

```
python postprocess.py 
python post_test.py 
```

### Note

When the modelname in `config.py` is `Process, Persistence, and w_climatology`, there is no need to run main.py to process data and training models.Just execute commands(`
python postprocess.py`and `python post_test.py`) for detailed analysis.

