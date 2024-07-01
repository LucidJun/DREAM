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

For demonstration purpose, we use the training pipeline of deep learning model and the  FACED dataset that is supported by TorchEEG `<https://torcheeg.readthedocs.io/en/latest/>`_. FACED dataset can be downloaded from the link `<https://www.synapse.org/#!Synapse:syn50614194/files/>`_. Online preprocessing
are applied as shown below after converting it into tensors and then fed to neural network as input. 






Setting up the pipeline for generating model card
-----------------------------------------------------


We initialize the TSception model and define its hyperparameters. Next, we configure the training and validation processes using PyTorch . 
After completing these steps, we employ a custom plotting function to generate and save plots for accuracy, loss, and the confusion matrix in the designated folder. 
Finally, we invoke the generate_model_card() function, which requires the path to the YAML config file, the output path for saving the model card, and the model card's
version number. The template of the config file is `here <https://github.com/LucidJun/DREAM/tree/main/template>`_.

.. code:: python

    from torch.utils.data import DataLoader
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from sklearn.metrics import ConfusionMatrixDisplay
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_score, recall_score,accuracy_score, f1_score, confusion_matrix
    import seaborn as sns




    from torch.utils.data import DataLoader, random_split
    from torcheeg.datasets import FACEDDataset
    from torcheeg import transforms
    from torcheeg.datasets.constants import FACED_CHANNEL_LOCATION_DICT

    from torcheeg.models.cnn import TSCeption
    import json
    import os
    import time
    from dreams_mc.make_model_card import generate_modelcard


    data_folder= "./processed_data_Face/Processed_data"

    dataset = FACEDDataset(root_path=data_folder,
                       online_transform=transforms.Compose(
                           [transforms.ToTensor(),
                            transforms.To2d()]),
                       label_transform=transforms.Compose([
                           transforms.Select('emotion'),
                           #transforms.Lambda(lambda x: x + 1)
                       ]))
    
    # dataloaders
    train_size = 0.8 
    batch_size = 32  

    num_train_samples = int(len(dataset) * train_size)
    num_val_samples = len(dataset) - num_train_samples

    # Split the dataset
    train_dataset, val_dataset = random_split(dataset, [num_train_samples, num_val_samples])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    #Initiate the model
    model = TSCeption(num_classes=9,
                  num_electrodes=30,
                  sampling_rate=250,
                  num_T=15,
                  num_S=15,
                  hid_channels=32,
                  dropout=0.5)

    device= torch.device( "cuda" if torch.cuda.is_available() else "cpu")

    #computing accuracy
    def compute_accuracy(y_pred, y_true):
        # Get the predicted class by selecting the maximum logit (log-probability)
        _, y_pred_tags = torch.max(y_pred, dim=1)
        
        # Compare predictions with true labels
        correct_pred = (y_pred_tags == y_true).float()
        
        # Compute accuracy
        acc = correct_pred.sum() / len(correct_pred)
        return acc

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.000001, max_lr=0.01)
    criterion = nn.CrossEntropyLoss()

    accuracy_stats = {
    'train': [],
    "val": []
    }
    loss_stats = {
    'train': [],
    "val": []
    }

    # Training  and validation function
     

    def train( n_epochs, val_acc_max_input ,model, optimizer, criterion, checkpoint_path, best_model_path,start_epoch=1):
        
        
        val_acc_max = val_acc_max_input 
        for e in tqdm(range(start_epoch, n_epochs+1)):
            
            # TRAINING
            train_epoch_loss = 0
            train_epoch_acc = 0
            model.train()
            for X_train_batch, y_train_batch in train_loader:

                X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
                optimizer.zero_grad()
                y_train_pred = model(X_train_batch)

                train_loss = criterion(y_train_pred, y_train_batch)
                train_acc = compute_accuracy(y_train_pred, y_train_batch)
                
                train_loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_epoch_loss += train_loss.item()
                train_epoch_acc += train_acc.item()
                
                
            # VALIDATION    
            with torch.no_grad():
                
                val_epoch_loss = 0
                val_epoch_acc = 0
                
                model.eval()
                for X_val_batch, y_val_batch in val_loader:
                    X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                    
                    y_val_pred = model(X_val_batch)
                                
                    val_loss = criterion(y_val_pred, y_val_batch)
                    val_acc = compute_accuracy(y_val_pred, y_val_batch)
                    
                    val_epoch_loss += val_loss.item()
                    val_epoch_acc += val_acc.item()

            loss_stats['train'].append(train_epoch_loss/len(train_loader))
            loss_stats['val'].append(val_epoch_loss/len(val_loader))
            accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
            accuracy_stats['val'].append(val_epoch_acc/len(val_loader))


            valid_accuracy= val_epoch_acc/len(val_loader)

            print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')


            # create checkpoint variable and add important data
            checkpoint = {
                    'epoch': e + 1,
                    'valid_acc_max': valid_accuracy,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
            save_ckp(checkpoint, False, checkpoint_path, best_model_path)

            if valid_accuracy > val_acc_max:
                    print('Validation accuracy increased ({:.6f} --> {:.6f}).  Saving model ...'.format(val_acc_max,valid_accuracy))
                    # save checkpoint as best model
                    save_ckp(checkpoint, True, checkpoint_path, best_model_path)
                    val_acc_max = valid_accuracy
                    
        return model
         
        # Initiate training 
        valid_acc_max=0.0
        trained_model = train(200, valid_acc_max, model, optimizer, criterion, "./current_checkpoint.pt", "./best_model.pt",start_epoch=1)

        #call plotting functions for loss and accuracy to save the plots
        plot_training_validation_stats(accuracy_stats, loss_stats,save_dir='./')

        # Call function to get the metrics
        results=evaluate_model(model=model,dataloader=val_loader,device=device)
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1 Score: {results['f1_score']:.4f}")
        print(f"Accuracy: {results['accuracy']:.4f}")

        # Plot the confusion matrix and save it 
        class_names = ['Class 0', 'Class 1','Class 2','Class 3','Class 4','Class 5','Class 6','Class 7']  # replace with your actual class names
        plot_confusion_matrix(results['confusion_matrix'], class_names,save_path="./cm.png")

        #Unvertainity estimation and save the plot
        plot_confidence_intervals(precision=results['precision'], recall=results['recall'], accuracy=results['accuracy'], f1=results['f1_score'], n=len(val_dataset),save_path="./CI_plot.png", confidence=0.95)

       # Call Model card function from dreams_mc

       # Path to your configuration file 
        config_file_path = './config.yaml'

        # Desired output path for the model card HTML file
        output_path = './model_card.html'

        # Version number of your model
        version_num = '1.0'

        # Generate the model card
        generate_modelcard(config_file_path, output_path, version_num)
        





        






