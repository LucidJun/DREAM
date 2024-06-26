[![Documentation Status](https://img.shields.io/readthedocs/dreams-mc)](https://dreams-mc.readthedocs.io/en/latest/index.html)
[![license](https://img.shields.io/badge/License-BSD%203-brightgreen)](https://github.com/LucidJun/DREAM/blob/main/LICENSE.txt)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/dreams_mc)](https://pypi.org/project/dreams-mc/)
[![Downloads](https://pepy.tech/badge/dreams_mc)](https://pepy.tech/project/dreams_mc)


# Deep Report for AI Models (DREAMS)

`dreams_mc` is a Python package designed to simplify the process of creating comprehensive model cards for deep learning models. Model cards(mc) are crucial for documenting the performance, usage, and limitations of AI models, promoting transparency and ethical use of AI.



## Features

- **Generate Comprehensive Model Cards**: Easily produce detailed model cards including sections like project overview, dataset description, results with plots, uncertainty estimation, and references.

- **Configurable**: Use a YAML configuration file to customize the model card content, making it adaptable to a wide range of projects and models.
- **Support for Various Plot Types**: Integrate visualizations such as performance graphs and uncertainty estimations directly into your model cards.

## Installation

You can install the `dreams` package using pip:

```python
pip install dreams_mc

```

Advanced users can take full potential of DREAMS by installing as a python package.

```python
pip install git+https://github.com/LucidJun/DREAM.git

```

To generate a model card, ensure you have a configuration file (config.yaml) prepared with the necessary details about your model, dataset, and results. Here's a simple example of generating a model card:

```python


from dreams_mc.make_model_card import generate_modelcard

# Path to your configuration file
config_file_path = '/path/to/your/config.yaml'

# Desired output path for the model card HTML file
output_path = '/path/to/output/model_card.html'

# Version number of your model
version_num = '1.0'

# Generate the model card
generate_modelcard(config_file_path, output_path, version_num)

```

## Configuration

The config.yaml file should contain sections for your project overview, dataset description, results, uncertainty estimation, and references. Here's a brief outline of what each section might include:

- Overview: General information about the model and its purpose.
- Dataset Description: Details about the dataset used, including preprocessing steps and split ratios.
- Results: Model performance metrics, possibly including training/validation losses, accuracy, etc.
- Uncertainty Estimation: Information on how uncertainty is quantified in your model's predictions.
- References: Citations or links to relevant papers, datasets, or other resources.




[## Citing]:#
[For citation]:#


[## Acknowledgements]: #
