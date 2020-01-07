# sits-classification
Classification of satellite image time series (SITS) using efficient time series classification methods

#### This repository is inspired by three observations: 
1. there is no openly available Sentinel-2 time series dataset that is ready for use with state-of-the-art
time series classification methods
2. publications about new time series classification methods typically evaluate their methods using a wide
 variety of datasets (e.g. UCR "bake-off" datasets), but seldomly satellite image time series datasets
3. these publications typically focus on achieved accuracy (rightly so) and training speed. However, they often
 do not evaluate prediction speeds in depth, which are of major interest for e.g. pixel-based land cover 
 classification tasks. For the production of land-cover classification maps, predictions need to be made for areas of 
 the size of whole countries which corresponds to hundreds of millions to billions of pixels/time series. 

#### This repository therefore...
1. establishes a Sentinel-2 SITS dataset from open-source data
2. evaluates time series classification algorithms on this Sentinel-2 SITS dataset, with the main focus on accuracy and prediction 
speed

#### Notebooks:
* 01_prepare_Sentinel-2_SITS_dataset.ipynb - establishes Sentinel-2 time series dataset
* 02_univariate_SITS_classification_ROCKET.ipynb - evaluates a new method for time series classification that uses 
convolutional kernels
