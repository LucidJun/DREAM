.. DREAMS_MC documentation master file, created by
   sphinx-quickstart on Wed May  8 11:51:05 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DREAMS_MC's documentation!
=====================================
.. image:: imgs/dreams_logo.png
    :align: center
    :scale: 35 %
    :alt: DREAMS_MC!

Deep Report for AI Models (DREAMS)
==================================

`dreams_mc` is a Python package designed to simplify the process of creating comprehensive model cards for deep learning models. Model cards(mc) are crucial for documenting the performance, usage, and limitations of AI models, promoting transparency and ethical use of AI.



Features
=========

- **Generate Comprehensive Model Cards**: Easily produce detailed model cards including sections like project overview, dataset description, results with plots, uncertainty estimation, and references.

- **Configurable**: Use a YAML configuration file to customize the model card content, making it adaptable to a wide range of projects and models.
- **Support for Various Plot Types**: Integrate visualizations such as performance graphs and uncertainty estimations directly into your model cards.

Installation
=============
The dreams_mc is hosted on PyPI. You can install the `dreams_mc` package using pip:

.. code:: bash

  $ pip install dreams_mc


Get started
===========

To generate a model card, ensure you have a configuration file (config.yaml) prepared with the necessary details about your model, dataset, and results. The plots/figures for publishing in the model card is stored in a folder. The folder path should be mentioned in the config file. Here's a simple example of generating a model card:

.. code:: python

   from dreams.make_model_card import generate_modelcard

   # Path to your configuration file
   config_file_path = '/path/to/your/config.yaml'

   # Desired output path for the model card HTML file
   output_path = '/path/to/output/model_card.html'

   # Version number of your model
   version_num = '1.0'

   # Generate the model card
   generate_modelcard(config_file_path, output_path, version_num)



Configuration
==============

The config.yaml file should contain sections for your project overview, dataset description, results, uncertainty estimation, and references. Here's a brief outline of what each section might include:

- Overview: General information about the model and its purpose.
- Dataset Description: Details about the dataset used, including preprocessing steps and split ratios.
- Results: Model performance metrics, possibly including training/validation losses, accuracy, etc.
- Uncertainty Estimation: Information on how uncertainty is quantified in your model's predictions.
- References: Citations or links to relevant papers, datasets, or other resources.



How to cite DREAMS_MC?
======================
.. admonition:: Note

   For citing us in your publication,
   Click :ref:`here<authors>` for further details.


Contents
========

.. toctree::
   :maxdepth: 2

   modules

   Get Started <descriptions/get_started>
   Attributes list for config file <descriptions/features>
   Frequently Asked Questions <descriptions/faq>
   Module Reference <descriptions/modules>
   Authors <authors>
   Changelog <changelog>
   License <license>




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`






