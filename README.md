[![Documentation Status](https://readthedocs.org/projects/dream/badge/?version=latest)](https://dream.readthedocs.io/en/latest/?badge=latest)
[![license](https://img.shields.io/badge/License-BSD%203-brightgreen)](https://github.com/pinnyin/dream/blob/master/LICENSE.txt)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/dream)
![PyPI](https://img.shields.io/pypi/v/dream)
[![Downloads](https://pepy.tech/badge/dream)](https://pepy.tech/project/dream)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pinnyin/dream/blob/master/notebooks/EEG_Example.ipynb)

# Deep Report for AI Models
## Training , Testing and Reporting
This repository hosts the **DREAM - Deep Report for AI Models** python package. DREAM assists researchers on exploratory feature extraction tasks on time series without requiring significant programming effort.

Users can interact with DREAM using two methods:
##### Online
It does not requires installation as it relies on Google Colabs and a user interface provided by Google Sheets

##### Offline
Advanced users can take full potential of DREAM by installing as a python package
```python
pip install dream
```

## Includes a comprehensive number of features
DREAM is optimized for time series and **automatically extracts over 60 different features on the statistical, temporal and spectral domains.**

## Functionalities
* **Intuitive, fast deployment and reproducible**: interactive UI for feature selection and customization
* **Computational complexity evaluation**: estimate the computational effort before extracting features
* **Comprehensive documentation**: each feature extraction method has a detailed explanation
* **Unit tested**: we provide unit tests for each feature
* **Easily extended**: adding new features is easy and we encourage you to contribute with your custom features

## Get started
The code below extracts all the available features on an example dataset file.

```python
import dream
import pandas as pd

# load dataset
df = pd.read_csv('Dataset.txt')

# Retrieves a pre-defined feature configuration file to extract all available features
cfg = dream.get_features_by_domain()

# Extract features
X = dream.time_series_features_extractor(cfg, df)
```

## Available features

#### Statistical domain
| Features                   | Computational Cost |
|----------------------------|:------------------:|
| Absolute energy            |          1         |
| Average power              |          1         |
| Entropy                    |          1         |
| Histogram                  |          1         |
| Interquartile range        |          1         |
| Kurtosis                   |          1         |
| Max                        |          1         |
| Mean                       |          1         |
| Mean absolute deviation    |          1         |
| Median                     |          1         |
| Median absolute deviation  |          1         |
| Min                        |          1         |
| Root mean square           |          1         |
| Skewness                   |          1         |
| Standard deviation         |          1         |
| Variance                   |          1         |





#### Spectral domain
| Features                          | Computational Cost |
|-----------------------------------|:------------------:|
| FFT mean coefficient              |          1         |
| Fundamental frequency             |          1         |
| Human range energy                |          2         |
| LPCC                              |          1         |
| MFCC                              |          1         |
| Max power spectrum                |          1         |
| Maximum frequency                 |          1         |
| Median frequency                  |          1         |
| Power bandwidth                   |          1         |
| Spectral centroid                 |          2         |
| Spectral decrease                 |          1         |
| Spectral distance                 |          1         |
| Spectral entropy                  |          1         |
| Spectral kurtosis                 |          2         |
| Spectral positive turning points  |          1         |
| Spectral roll-off                 |          1         |
| Spectral roll-on                  |          1         |
| Spectral skewness                 |          2         |
| Spectral slope                    |          1         |
| Spectral spread                   |          2         |
| Spectral variation                |          1         |
| Wavelet absolute mean             |          2         |
| Wavelet energy                    |          2         |
| Wavelet standard deviation        |          2         |
| Wavelet entropy                   |          2         |
| Wavelet variance                  |          2         |


## Citing
For citation:

Khadka, ... et al. "*DREAM: Deep Report .*" Software(20). [https://doi.org/106/j.so.16](https://doi.org/106/j.so.16)

## Acknowledgements
This project...
