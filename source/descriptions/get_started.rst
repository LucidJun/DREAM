===========
Get Started
===========

Overview
--------

Given a dataset, we perform the exploratory data analysis (EDA). We store the visualization plots into a folder which will be further populated with the plots (loss, accuracy, confusion matrix e.t.c ) after model´s training and validation process is completed. The attributes The pipeline for generating the model card is shown in the fiugure below.

.. image:: ../imgs/pipeline.png
    :align: center
    :scale: 70 %
    :alt: DREAMS!

Start with the dataset
-------------------------

For demonstration purpose, we use the training pipeline of deep learning model and the  DEAP dataset that is supported by TorchEEG `<https://torcheeg.readthedocs.io/en/latest/>`_. DEAP dataset can be downloaded from the link `<https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html>`_. Online preprocessing
are applied as shown below after converting it into tensors and then fed to neural network as input. 

.. code:: python

    from dreams_mc import generate_modelcard
    from torcheeg.datasets import DEAPDataset
    from torcheeg import transforms
    from torcheeg.datasets.constants import DEAP_CHANNEL_LOCATION_DICT
    from torch.utils.data import DataLoader 
    from torcheeg.models import CCNN 
    from torcheeg.trainers import ClassifierTrainer
    import pytorch_lightning as pl
    import numpy as np

    dataset = DEAPDataset(
    io_path=f'./examples_mc/deap',
    root_path='./data_preprocessed_mc',
    offline_transform=transforms.Compose([
        transforms.BandDifferentialEntropy(apply_to_baseline=True),
        transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT, apply_to_baseline=True)
    ]),
    online_transform=transforms.Compose(
        [transforms.BaselineRemoval(),
         transforms.ToTensor()]),
    label_transform=transforms.Compose([
        transforms.Select('valence'),
        transforms.Binary(5.0),
    ]),
    num_worker=8)


.. code:: python

    # Split the Dataset into Training and Test Sets
    from torcheeg.model_selection import KFoldGroupbyTrial

     
    k_fold = KFoldGroupbyTrial(n_splits=10,
                           split_path='./examples_mc/split',
                           shuffle=True,
                           random_state=42)


Setting up the pipeline for generating model card
-----------------------------------------------------


We initialize the TSception model and define its hyperparameters. Next, we configure the training and validation processes using PyTorch Lightning's "fit" method. 
After completing these steps, we employ a custom plotting function to generate and save plots for accuracy, loss, and the confusion matrix in the designated folder. 
Finally, we invoke the generate_model_card() function, which requires the path to the YAML config file, the output path for saving the model card, and the model card's
version number. The template of the config file is `here <https://github.com/LucidJun/DREAM/tree/main/template>`_.

.. code:: python

    from torch.utils.data import DataLoader
    from torcheeg.models.cnn import TSCeption

    from torcheeg.trainers import ClassifierTrainer

    import pytorch_lightning as pl


    for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):
      train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
      val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

      model = TSCeption(num_classes=2,
                    num_electrodes=28,
                    sampling_rate=128,
                    num_T=15,
                    num_S=15,
                    hid_channels=32,
                    dropout=0.5)

      trainer = ClassifierTrainer(model=model,
                                  num_classes=2,
                                  lr=1e-4,
                                  weight_decay=1e-4,
                                  accelerator="gpu")
      trainer.fit(train_loader,
                  val_loader,
                  max_epochs=50,
                  default_root_dir=f'./examples_mc/model/{i}',
                  callbacks=[pl.callbacks.ModelCheckpoint(save_last=True)],
                  enable_progress_bar=True,
                  enable_model_summary=True,
                  limit_val_batches=0.0)
      score = trainer.test(val_loader,
                          enable_progress_bar=True,
                          enable_model_summary=True)[0]
      print(f'Fold {i} test accuracy: {score["test_accuracy"]:.4f}')





