List of available attributes in config file
=============================================

The list of attributes correpsonds to the configurable variables in the 'config.yaml'. The path of the 'config.yaml' file should be given as the one of the input parameters for genreat_modelcard(). The table below outlines the attributes that can be filled in the model card for the current version of dreams_mc.
See the template of the config file `here <https://github.com/LucidJun/DREAM/tree/main/template>`_.


.. autosummary::
   :toctree: _generated
   :template: module_functions_template.rst



====================          ============================================================================
Attribute Name                 Description                                                                                                                                                                                                                                         
====================          ============================================================================
model_version                  The version number of the model card.
logo_path                      The path to the logo of the project or organisation.         
dataset_name                   Name of the dataset used for training the model.       
num_target_class               Number of classes in the dataset under supervised setting.                
ground_truth                   Target label for training and validating the model under supervised setting.
split_ratio                    The ratio for splitting dataset into training and validation and test set.
preprocess_steps               Names of the preprocess steps applied to the data.
model_type                     Name of the model/architecture applied.
model_input                    Input to the model (as a list if many input types) .
model_output                   Model´s Output (as a list if many).
learning_rate                  Learning rate used in training.
batch_size                     Batch size of data .
additional_info                Additional information about the model (if required).
describe_overview              Overview of the  model report.
describe_dataset               Description of the dataset used.
model_details                  Description of the applied model.
limitation_details             Describing limitations about the model.
performance_comments           Describing about the model´s performance.
uncertainty_describe           Describing the uncertainty of the model´s performance.
data_figpath                   Path to the data distribution figure (obtained from exploratory data analysis)
loss_figpath                   Path to the training and validation loss figure(obtained after model training and validation)
acc_figpath                    Path to the training and validatioin accuracy figure (obtained after model training and validation)
cm_figpath                     Path to the confusion matrix (obtained after model training and validation)
uncertainty_figpath            Path to the figure depicting uncertainty estimation (obtained after model training and validation )
result_table_figpath           Path to the figure for result table (if necessary to display).

====================          ============================================================================
                              

 