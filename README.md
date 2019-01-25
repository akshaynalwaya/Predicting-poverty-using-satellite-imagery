# CSC591: Combining satellite imagery and machine learning to predict poverty

The code in this repository is an attempt to come up with a new Convolutional Neural Network model to predict poverty of nations in Africa by using satellite imagery and night-time light intensities. Our goal is to achieve a similar performance shown by Jean, Burke, et al. (2016) [link](https://github.com/nealjean/predicting-poverty)

### Description of folders

- **data**: Input DHS and LSMS data, Nightlight, Intensity text files and training images
- **scripts**: Scripts used to process survey data and acquire training satellite images.
- **cnn-model**: Store parameters for trained convolutional neural network and scripts for generating the model and predictions

### Packages and Tools required
Code was written in R 3.4.1 and Python 2.7.14

**Python**
- Jupyter
- requests
- NumPy
- Pandas
- SciPy
- scikit-learn
- Seaborn
- OpenCV 3.1.0

We suggest using [Anaconda](https://www.continuum.io/downloads).

The user can run the following command to automatically install the python packages after installing anaconda.
```
conda install jupyter requests numpy pandas scipy scikit-learn seaborn
```

Install Tensorflow v1.3.0 [Install Instructions](https://www.tensorflow.org/install/) with GPU
Keras [Install Instructions](https://keras.io/#installation) with GPU

**R**
- R.utils
- magrittr
- foreign
- raster
- readstata13
- plyr
- dplyr
- sp
- rgdal

Install RStudio v1.1.423

The user can run the following command to automatically install the R packages
```
install.packages(c('R.utils', 'magrittr', 'foreign', 'raster', 'readstata13', 'plyr', 'dplyr', 'sp', 'rgdal'), dependencies = T)
```

### Instructions for processing survey data

Due to data access agreements, users need to independently download data files from the World Bank's Living Standards Measurement Surveys and the Demographic and Health Surveys websites. These two data sources require the user to fill in a Data User Agreement form and register for an account.

1. Download DHS data
	1. Visit the [host website for the Demographic and Health Surveys data](http://dhsprogram.com/data/dataset_admin/download-datasets.cfm)
	2. Download survey data into **data/input/DHS**. The relevant data are from the Standard DHS surveys corresponding to the following country-years:
		1. Uganda 2011
		2. Tanzania 2010
		3. Rwanda 2010
		4. Nigeria 2013
		5. Malawi 2010
	3. For each survey, the user should download its corresponding Household Recode files in Stata format as well as its corresponding geographic datasets
	4. Unzip these files so that **data/input/DHS** contains the following folders of data: (Note that the names of these folders may vary slightly depending on the date the data is downloaded)
		1. UG_2011_DHS_03022018_1051_118965
		2. TZ_2010_DHS_03022018_1051_118965
		3. RW_2010_DHS_03022018_1053_118965
		4. NG_2013_DHS_03022018_1053_118965
		5. MW_2010_DHS_03022018_1053_118965

2. Download LSMS data
	1. Visit the [website for the World Bank's LSMS data](http://microdata.worldbank.org/index.php/catalog/lsms):
	2. Download into **data/input/LSMS** the files corresponding to the following country-years:
 		1. Uganda 2011-2012
		2. Tanzania 2012-13
		3. Nigeria 2012-13
		4. Malawi 2013

	3. Unzip these files so that **data/input/LSMS** contains the following folders of data:
		1. UGA_2011_UNPS_v01_M_STATA
		2. TZA_2012_LSMS_v01_M_STATA_English_labels
		3. DATA (formerly NGA_2012_LSMS_v03_M_STATA before a re-upload in January 2016)
		4. MWI_2013_IHPS_v01_M_STATA


3. Run the following files in the script folder
  1. Set working directory as the repository
  2. DownloadNightLightData.R
  3. PreProcess_DHS_Data.R
	4. PreProcess_LSMS_Data.R

### Instructions to download satellite images used to train our convolutional neural network

1. Run scripts/get_coordinates.R
2. Set the working directory as the 'scripts' folder in the repository
3. Run the following jupyter notebooks (this process could take 2-3 hours to execute)
  1. Classification_XYZ.ipynb
  2. getting_images.ipynb

Now your **data/images** folder should be populated with satellite images.

### Instructions for using our cnn model for training and predictions

##### Creating training data from **data/images**.

1. Create folders names 'class1', 'class2' and 'class3' in **cnn-model**
2. Take the images of class1 category of all countries in **data/images** and copy them to the **cnn-model/class1** folder.
3. Repeat the 2nd step for class2 and class3 as well.

##### Training the model and predictions

1. set working directory to cnn-model
The model has been trained for epochs = 100 and batch_size = 32.
2. run train.py
	1. This will create a hdf5 object file with weights of the trained cnn model.
	2. This will also create a model.JSON file with the model architecture.
3. Run predict.py

##### Perform predictions with our pre-trained model

1. Download the hdf5 object and model.JSON files from [here](https://drive.google.com/drive/folders/1f0xSwNM56ZSw7pe9gZuO3iXzInIsAXzk?usp=sharing)
2. Save it in cnn-model folder.
3. Run predict.py
