"""Trainer for DeepPhys Model"""

import os

import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.model.DeepPhys import DeepPhys
from neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm

class DeepPhysTrainer(BaseTrainer):
    def __init__(self, config, data_loader):
        """
        Initializes the DeepPhysTrainer with configuration settings and data loader.

        This constructor sets up the training environment by initializing the device,
        model, optimizer, and learning rate scheduler based on the provided configuration.
        It also prepares the model for training or testing based on the specified mode.

        Args:
            config: A configuration object containing various settings for training/testing.
            data_loader: A dictionary containing data loaders for training, validation, and testing.

        Raises:
            ValueError: If the toolbox mode is not recognized.
        """

        # Initialize the base class
        super().__init__()

        # Set the device (CPU or GPU) for model training/testing
        self.device = torch.device(config.DEVICE)

        # Retrieve training configuration parameters
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        self.config = config

        # Initialize variables for tracking validation loss and best epoch
        self.min_valid_loss = None
        self.best_epoch = 0
        
        # Check the mode of operation: train and test or only test
        if config.TOOLBOX_MODE == "train_and_test":
            # Initialize the DeepPhys model with specified image size and Enable multi-GPU training
            self.model = DeepPhys(img_size=config.TRAIN.DATA.PREPROCESS.RESIZE.H).to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))

            # Calculate the number of training batches
            self.num_train_batches = len(data_loader["train"])

            # Set the loss function to Mean Squared Error (MSE) Loss
            self.criterion = torch.nn.MSELoss()

            # Initialize the AdamW optimizer with learning rate and weight decay
            self.optimizer = optim.AdamW(self.model.parameters(), 
                                         lr=config.TRAIN.LR, 
                                         weight_decay=0)
            
            # Set up the OneCycleLR scheduler for dynamic learning rate adjustment. Info: https://tinyurl.com/j65ywsut
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                                 max_lr=config.TRAIN.LR, 
                                                                 epochs=config.TRAIN.EPOCHS, 
                                                                 steps_per_epoch=self.num_train_batches)
        
        elif config.TOOLBOX_MODE == "only_test":
            # Initialize the DeepPhys model for testing and Enable multi-GPU testing
            self.model = DeepPhys(img_size=config.TEST.DATA.PREPROCESS.RESIZE.H).to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
        else:
            # Raise an error if the toolbox mode is incorrect
            raise ValueError("DeepPhys trainer initialized in incorrect toolbox mode!")

    def train(self, data_loader):
        """Training routine for model"""

        # Check if training data is available
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        # Initialize lists to store mean losses and learning rates
        mean_training_losses = []
        mean_valid_losses = []
        lrs = []

        # Loop over each epoch
        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_loss = []

            # Set model to training mode
            self.model.train()
            
            # Iterate over training data batches with a progress bar
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)

                # Move data and labels to the specified device
                data, labels = batch[0].to(self.device), batch[1].to(self.device)

                # Reshape data to match model input dimensions
                N, D, C, H, W = data.shape
                data = data.view(N * D, C, H, W)
                labels = labels.view(-1, 1)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass to get predictions
                pred_ppg = self.model(data)

                # Compute the loss
                loss = self.criterion(pred_ppg, labels)

                # Backward pass to compute gradients
                loss.backward()

                # Append the current learning rate to the list
                lrs.append(self.scheduler.get_last_lr())

                # Update model weights
                self.optimizer.step()

                # Update learning rate scheduler
                self.scheduler.step()

                # Accumulate running loss
                running_loss += loss.item()

                # Print running loss every 100 mini-batches
                if idx % 100 == 99:
                    print(f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0

                # Append current batch loss to train_loss
                train_loss.append(loss.item())

                # Update progress bar with current loss and learning rate
                tbar.set_postfix({"loss": loss.item(), "lr": self.optimizer.param_groups[0]["lr"]})

            # Calculate and store mean training loss for the epoch
            mean_training_losses.append(np.mean(train_loss))

            # Save model state for the current epoch
            self.save_model(epoch)

            # Run validation and use the best epoch for testing if enabled
            if self.config.VALID.RUN_VALIDATION and self.config.TEST.USE_BEST_EPOCH: 
                valid_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                print('validation loss: ', valid_loss)

                # Update best model based on validation loss
                if self.min_valid_loss is None:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))

                elif (valid_loss < self.min_valid_loss):
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
        
        # Print best epoch and minimum validation loss if condition met
        if self.config.VALID.RUN_VALIDATION and self.config.TEST.USE_BEST_EPOCH: 
            print("best trained epoch: {}, min_val_loss: {}".format(self.best_epoch, self.min_valid_loss))
        
        # Plot losses and learning rates if configured
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)

    def valid(self, data_loader):
        """Model evaluation on the validation dataset"""
    
        # Check if validation data is available
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        # Print start of validation process
        print('')
        print("===Validating===")

        # Initialize list to store validation losses
        valid_loss = []

        # Set model to evaluation mode
        self.model.eval()

        # Initialize validation step counter
        valid_step = 0

        # Disable gradient computation for validation
        with torch.no_grad():
            # Create a progress bar for the validation loop
            vbar = tqdm(data_loader["valid"], ncols=80)

            # Iterate over validation batches
            for valid_idx, valid_batch in enumerate(vbar):
                # Set description for the progress bar
                vbar.set_description("Validation")

                # Move data and labels to the appropriate device
                data_valid, labels_valid = valid_batch[0].to(self.device), valid_batch[1].to(self.device)
                
                # Reshape data to match model input dimensions
                N, D, C, H, W = data_valid.shape
                data_valid = data_valid.view(N * D, C, H, W)
                labels_valid = labels_valid.view(-1, 1)

                # Make predictions using the model
                pred_ppg_valid = self.model(data_valid)

                # Calculate loss between predictions and actual labels
                loss = self.criterion(pred_ppg_valid, labels_valid)

                # Append loss to the list of validation losses
                valid_loss.append(loss.item())

                # Increment validation step counter
                valid_step += 1

                # Update progress bar with current loss
                vbar.set_postfix(loss=loss.item())

            # Convert list of losses to a NumPy array
            valid_loss = np.asarray(valid_loss)

        # Return the mean of the validation losses
        return np.mean(valid_loss)

    def test(self, data_loader):
        """Model evaluation on the testing dataset"""
        
        # Check if test data is available
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        
        config = self.config
        print('')
        print("===Testing===")

        # Initialize dictionaries to store predictions and true labels
        predictions = dict()
        labels = dict()

        # Load the appropriate model based on the configuration
        if self.config.TOOLBOX_MODE == "only_test":
            # Load pre-trained model for inference
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")

        else:
            # Load model from the best epoch based on configuration
            if self.config.TEST.USE_BEST_EPOCH:
                best_model_path = os.path.join(self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path))

            # Load model from the last epoch based on configuration
            else:
                last_epoch_model_path = os.path.join(self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path))

        # Set model to evaluation mode
        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        print("Running model evaluation on the testing dataset!")

        # Evaluate the model on the test dataset
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                batch_size = test_batch[0].shape[0]

                # Move data and labels to the appropriate device
                data_test, labels_test = test_batch[0].to(self.config.DEVICE), test_batch[1].to(self.config.DEVICE)

                # Reshape data and labels for model input
                N, D, C, H, W = data_test.shape
                data_test = data_test.view(N * D, C, H, W)
                labels_test = labels_test.view(-1, 1)

                # Get model predictions
                pred_ppg_test = self.model(data_test)

                # Move predictions and labels to CPU if output saving is enabled
                if self.config.TEST.OUTPUT_SAVE_DIR:
                    labels_test = labels_test.cpu()
                    pred_ppg_test = pred_ppg_test.cpu()

                # Store predictions and labels in dictionaries
                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    labels[subj_index][sort_index] = labels_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]
        
        print('')
        
        # Calculate and print evaluation metrics
        calculate_metrics(predictions, labels, self.config)

        # Save test outputs if specified in the configuration
        if self.config.TEST.OUTPUT_SAVE_DIR: # saving test outputs
            self.save_test_outputs(predictions, labels, self.config)

    def save_model(self, index):
        """
        Saves the current state of the model to a file.

        Parameters:
        index (int): The current epoch number, used to differentiate saved model files.
        """
        # Check if the model directory exists; if not, create it
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Construct the file path for saving the model
        model_path = os.path.join(self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')

        # Save the model's state dictionary to the specified file path
        torch.save(self.model.state_dict(), model_path)