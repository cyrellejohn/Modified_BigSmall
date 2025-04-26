"""Trainer for BigSmall Multitask Models"""

# Training / Eval Imports 
import torch
import torch.optim as optim
import re

from neural_methods.model.BigSmall import BigSmall
from neural_methods.trainer.BaseTrainer import BaseTrainer

from evaluation.bigsmall_multitask_metrics import (calculate_ppg_metrics, calculate_openface_au_metrics)

'''
from neural_methods import loss
'''

# Other Imports
from collections import OrderedDict
import numpy as np
import os
from tqdm import tqdm

class BigSmallTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        """
        Initialize the BigSmallTrainer for multitask learning.

        Args:
            config: Configuration object containing model and training parameters.
            data_loader: DataLoader object for loading training and validation data.
        """
        # Print initialization message
        print('')
        print('Initializing BigSmall Multitask Trainer\n\n')

        # Save the configuration object for later use
        self.config = config
        self.use_scaler = False # TODO: WILL FULLY IMPLEMENT IN THE FUTURE
        label_list_path = os.path.dirname(config.TRAIN.DATA.DATA_PATH)
        self.num_train_batches = len(data_loader["train"])

        # Set up the compute device (GPU or CPU)
        if torch.cuda.is_available() and config.NUM_OF_GPU_TRAIN > 0:
            # Use the specified GPU if available
            self.device = torch.device(config.DEVICE) 
            self.num_of_gpu = config.NUM_OF_GPU_TRAIN
            self.using_TSM = True # Flag indicating the use of Temporal Shift Module
        else:
            # Default to CPU if no GPUs are available
            self.device = "cpu"
            self.num_of_gpu = 0
            self.using_TSM = False # Flag indicating the use of Temporal Shift Module

        # Define the model and set the device
        self.model = self.define_model(config)

        # Enable data parallelism if multiple GPUs are available
        if torch.cuda.device_count() > 1 and config.NUM_OF_GPU_TRAIN > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))

        # Send the model to the specified device
        self.model = self.model.to(self.device)

        # Retrieve training parameters from the configuration
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.LR = config.TRAIN.LR

        """
        AU_weights = torch.as_tensor([9.64, 11.74, 16.77, 1.05, 0.53, 0.56,
                                    0.75, 0.69, 8.51, 6.94, 5.03, 25.00]).to(self.device)
        self.criterionAU = torch.nn.BCEWithLogitsLoss(pos_weight=AU_weights).to(self.device)
        """
        
        # Set up loss functions for different tasks
        self.criterionAU = torch.nn.BCEWithLogitsLoss().to(self.device)
        self.criterionPPG = torch.nn.MSELoss().to(self.device)
        self.criterionEmotion = torch.nn.CrossEntropyLoss().to(self.device)

        # Initialize the optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.LR, weight_decay=0)

        # Initialize GradScaler only if on GPU
        if torch.cuda.is_available() and self.use_scaler:
            self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None # Loss scalar

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.LR, 
                                                             epochs=self.max_epoch_num, 
                                                             steps_per_epoch=self.num_train_batches)
            
        # Set up model saving information
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH

        # Initialize the epoch to use for testing
        self.used_epoch = 0

        # Define the target AU labels to be used for training
        target_aus = ['AU1_lf', 'AU2_lf', 'AU4_lf', 'AU6_lf', 'AU7_lf', 
                      'AU10_lf', 'AU12_lf', 'AU14_lf', 'AU15_lf', 'AU17_lf', 
                      'AU23_lf', 'AU24_lf']
        target_emotion = ['emotion_lf']

        # Define label lists and calculate indices for used labels
        label_list = self.load_label_names(label_list_path)

        au_label_list = [label for label in label_list if label in target_aus]
        ppg_label_list_train = [label for label in label_list if label == 'pos_ppg']
        ppg_label_list_test = [label for label in label_list if label == 'ppg'] # Option: ppg or filtered_ppg
        emotion_label_list = [label for label in label_list if label in target_emotion]

        self.au_label_names = au_label_list

        self.label_idx_train_au = self.get_label_idxs(label_list, au_label_list)
        self.label_idx_valid_au = self.get_label_idxs(label_list, au_label_list)
        self.label_idx_test_au = self.get_label_idxs(label_list, au_label_list)

        self.label_idx_train_ppg = self.get_label_idxs(label_list, ppg_label_list_train)
        self.label_idx_valid_ppg = self.get_label_idxs(label_list, ppg_label_list_train)
        self.label_idx_test_ppg = self.get_label_idxs(label_list, ppg_label_list_test)

        self.label_idx_train_emotion = self.get_label_idxs(label_list, emotion_label_list)
        self.label_idx_valid_emotion = self.get_label_idxs(label_list, emotion_label_list)
        self.label_idx_test_emotion = self.get_label_idxs(label_list, emotion_label_list)

    def load_label_names(self, label_list_path):
        full_path = os.path.join(label_list_path, "label_list.txt")

        with open(full_path, "r") as file:
            label_list = [line.strip() for line in file]  # Read and strip spaces

        return label_list

    def get_label_idxs(self, label_list, used_labels):
        label_idxs = []
        for l in used_labels:
            idx = label_list.index(l)
            label_idxs.append(idx)
        return label_idxs

    def define_model(self, config):
        """
        Initializes and configures the BigSmall model.

        Parameters:
        - config: Configuration object containing model parameters.

        Returns:
        - model: An instance of the BigSmall model configured based on the provided settings.
        """
        
        # Initialize the BigSmall model with a specified number of segments
        model = BigSmall(n_segment=3)
        
        # If using the Temporal Shift Module (TSM), configure additional parameters
        if self.using_TSM:
            # Set the frame depth from the configuration
            self.frame_depth = config.MODEL.BIGSMALL.FRAME_DEPTH
            # Calculate the base length as the product of the number of GPUs and frame depth
            self.base_len = self.num_of_gpu * self.frame_depth 

        # Return the configured model
        return model

    def format_data_shape(self, data, labels):
        """
        Reshapes and formats the input data and labels for model training or evaluation.

        Args:
            data (list): A list containing two tensors, `big_data` and `small_data`, representing different scales of input data.
            labels (torch.Tensor): A tensor containing the labels corresponding to the input data.

        Returns:
            tuple: A tuple containing the reshaped data list and labels tensor.
        """

        # Extract and reshape the Big data
        big_data = data[0]
        B, T, C, H, W = big_data.shape
        big_data = big_data.view(B * T, C, H, W)

        # Extract and reshape the Small data
        small_data = data[1]
        B, T, C, H, W = small_data.shape
        small_data = small_data.view(B * T, C, H, W)

        # Ensure labels have three dimensions and reshape them
        if len(labels.shape) != 3: 
            labels = torch.unsqueeze(labels, dim=-1)

        # The expected shape of labels is (B_label, T_label, C_label), where:
        # - B_label is the batch size
        # - T_label is the number of chunks
        # - C_label is the number of label classes
        B_label, T_label, C_label = labels.shape 

        # Reshape labels to flatten the first two dimensions
        labels = labels.view(B_label * T_label, C_label) 

        # Adjust data and labels for Temporal Shift Module (TSM) if used
        if self.using_TSM:
            # Ensure data and labels are multiples of base_len
            big_data = big_data[:(B * T) // self.base_len * self.base_len]
            small_data = small_data[:(B * T) // self.base_len * self.base_len]
            labels = labels[:(B * T) // self.base_len * self.base_len]

        # Update the data list with reshaped tensors
        data[0] = big_data
        data[1] = small_data

        # Add an extra dimension to labels to ensure correct shape
        labels = torch.unsqueeze(labels, dim=-1)

        # Return the formatted data and labels
        return data, labels
    
    def send_data_to_device(self, data, labels):
        """
        Transfers the input data and labels to the specified computing device (GPU or CPU).

        Args:
            data (tuple): A tuple containing two tensors, `big_data` and `small_data`, representing different scales of input data.
            labels (torch.Tensor): A tensor containing the labels corresponding to the input data.

        Returns:
            tuple: A tuple containing the data and labels, both transferred to the specified device.
        """
        # Transfer the Big data tensor to the specified device
        big_data = data[0].to(self.device)

        # Transfer the Small data tensor to the specified device
        small_data = data[1].to(self.device)

        # Transfer the labels tensor to the specified device
        labels = labels.to(self.device)

        # Pack the transferred data tensors back into a tuple
        data = (big_data, small_data)

        # Return the data and labels, both on the specified device
        return data, labels

    def save_model(self, index):
        """
        Saves the current state of the model to a file.

        Args:
            index (int): The epoch index used to uniquely identify the saved model file.
        """
        # Check if the model directory exists and create it if it doesn't
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Construct the file path for saving the model
        model_path = os.path.join(self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')

        # Save the model's state dictionary to the constructed file path
        torch.save(self.model.state_dict(), model_path)

        # Print a confirmation message with the saved model path
        print('Saved Model Path: ', model_path)
        print('')

    def train(self, data_loader):
        """
        Trains the BigSmall multitask model using the provided data loader.

        Args:
            data_loader (dict): A dictionary containing data loaders for training and validation datasets.
                                The key "train" should map to the training data loader, and "valid" to the validation data loader.

        Raises:
            ValueError: If the training data loader is not provided in the data_loader dictionary.

        Description:
            This method performs the training routine for the BigSmall model over a specified number of epochs.
            It initializes and tracks various loss metrics for different tasks (AU, PPG, Emotion) and updates the model's
            weights using backpropagation. The method also handles model saving and validation, updating the best model
            based on validation loss. If configured, it plots the training and validation losses along with learning rates.

        Returns:
            None
        """

        # Check if training data is available
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        print('Starting Training Routine')
        print('')

        # Initialize minimum validation loss as infinity
        min_valid_loss = np.inf

        # Initialize dictionaries to store training and validation losses for each task
        train_loss_dict = dict()
        train_au_loss_dict = dict()
        train_ppg_loss_dict = dict()
        train_emotion_loss_dict = dict()

        val_loss_dict = dict()
        val_au_loss_dict = dict()
        val_ppg_loss_dict = dict()
        val_emotion_loss_dict = dict()

        # TODO: Expand tracking and subsequent plotting of these losses for BigSmall
        mean_training_losses = []
        mean_valid_losses = []
        lrs = []

        # Iterate through each epoch
        for epoch in range(self.max_epoch_num):
            print(f"====Training Epoch: {epoch}====")

            # Initialize parameters for tracking losses
            running_loss = 0.0 
            train_loss = []
            train_au_loss = []
            train_ppg_loss = []
            train_emotion_loss = []

            self.model.train() # Set model to training mode
            
            # Iterate over training data batches
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)

                # Gather and format batch data
                data, labels = batch[0], batch[1]
                data, labels = self.format_data_shape(data, labels)
                data, labels = self.send_data_to_device(data, labels)

                # Forward and backward propagation
                self.optimizer.zero_grad()
                au_output, ppg_output, emotion_output = self.model(data)
                
                au_loss = self.criterionAU(au_output, labels[:, self.label_idx_train_au, 0])
                ppg_loss = self.criterionPPG(ppg_output, labels[:, self.label_idx_train_ppg, 0])
                emotion_loss = self.criterionEmotion(emotion_output, labels[:, self.label_idx_train_emotion, 0])

                loss = au_loss + ppg_loss + emotion_loss
                loss.backward()

                # Track learning rate
                lrs.append(self.scheduler.get_last_lr())

                # Update model weights
                self.optimizer.step()

                if torch.cuda.is_available() and self.use_scaler:
                    self.scaler.scale(loss).backward() # Loss scaling
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                # Update running loss and append losses to lists
                train_au_loss.append(au_loss.item())
                train_ppg_loss.append(ppg_loss.item())
                train_emotion_loss.append(emotion_loss.item())
                train_loss.append(loss.item())

                running_loss += loss.item()
                
                # Print every 100 mini-batches
                if idx % 100 == 99: 
                    print(f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0

                # Update progress bar with current loss and learning rate
                tbar.set_postfix({"loss:": loss.item(), "lr:": self.optimizer.param_groups[0]["lr"]})

            # Store epoch losses in dictionaries
            train_loss_dict[epoch] = train_loss
            train_au_loss_dict[epoch] = train_au_loss
            train_ppg_loss_dict[epoch] = train_ppg_loss
            train_emotion_loss_dict[epoch] = train_emotion_loss
            
            print('')

            # Append mean training loss for the epoch
            mean_training_losses.append(np.mean(train_loss))

            # Save model for the current epoch
            self.save_model(epoch)

            # Perform validation if enabled
            if self.config.VALID.RUN_VALIDATION:
                valid_loss, valid_au_loss, valid_ppg_loss, valid_emotion_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                val_loss_dict[epoch] = valid_loss
                val_au_loss_dict[epoch] = valid_au_loss
                val_ppg_loss_dict[epoch] = valid_ppg_loss
                val_emotion_loss_dict[epoch] = valid_emotion_loss
                print('Validation Loss: ', valid_loss)

                # Update the best model based on validation loss
                if self.config.TEST.USE_BEST_EPOCH and (valid_loss < min_valid_loss):
                    min_valid_loss = valid_loss
                    self.used_epoch = epoch
                    print("Updating Best model | Best epoch: {}".format(self.used_epoch))
                    print("Best model epoch:{}, val_loss:{}".format(self.used_epoch, min_valid_loss))

                else:
                    self.used_epoch = epoch
                    print("Model trained epoch:{}, val_loss:{}".format(self.used_epoch, min_valid_loss))
            
            else: 
                self.used_epoch = epoch
                print("Model trained epoch:{}, val_loss:{}".format(self.used_epoch, min_valid_loss))

            print('')
        
        # Plot losses and learning rates if configured
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)

        print('')

    def valid(self, data_loader):
        """
        Evaluates the BigSmall multitask model on the validation dataset.

        Args:
            data_loader (dict): A dictionary containing data loaders for training and validation datasets.
                                The key "valid" should map to the validation data loader.

        Raises:
            ValueError: If the validation data loader is not provided in the data_loader dictionary.

        Description:
            This method performs the evaluation routine for the BigSmall model using the validation dataset.
            It initializes and tracks various loss metrics for different tasks (AU, PPG, Emotion) without updating
            the model's weights. The method calculates the average loss for each task and the overall loss across
            all validation batches.

        Returns:
            tuple: A tuple containing the mean validation loss, mean AU loss, mean PPG loss, and mean Emotion loss
        """

        # Check if validation data is available
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        # Print a message indicating the start of validation
        print("===Validating===")

        # Initialize lists to store validation losses for each task
        valid_loss = []
        valid_au_loss = []
        valid_ppg_loss = []
        valid_emotion_loss = []

        # Set the model to evaluation mode
        self.model.eval()

        # Perform model validation without tracking gradients
        with torch.no_grad():
            # Create a progress bar for the validation data loader
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                # Update the progress bar description
                vbar.set_description("Validation")

                # Extract and format batch data and labels
                data, labels = valid_batch[0], valid_batch[1]
                data, labels = self.format_data_shape(data, labels)
                data, labels = self.send_data_to_device(data, labels)

                # Perform a forward pass to get model outputs
                au_out, ppg_out, emotion_out = self.model(data)

                # Calculate losses for each task
                au_loss = self.criterionAU(au_out, labels[:, self.label_idx_valid_au, 0]) 
                ppg_loss = self.criterionPPG(ppg_out, labels[:, self.label_idx_valid_ppg, 0]) 
                emotion_loss = self.criterionEmotion(emotion_out, labels[:, self.label_idx_valid_emotion, 0])

                # Sum the losses to get the total loss for the batch
                loss = au_loss + ppg_loss + emotion_loss

                # Append the individual and total losses to their respective lists
                valid_au_loss.append(au_loss.item())
                valid_ppg_loss.append(ppg_loss.item())
                valid_emotion_loss.append(emotion_loss.item())
                valid_loss.append(loss.item())

                # Update the progress bar with the current loss
                vbar.set_postfix(loss=loss.item())

        # Convert loss lists to numpy arrays and calculate mean losses
        valid_au_loss = np.asarray(valid_au_loss)
        valid_ppg_loss = np.asarray(valid_ppg_loss)
        valid_loss = np.asarray(valid_loss)
        valid_emotion_loss = np.asarray(valid_emotion_loss)

        # Return the mean total validation loss and individual task losses
        return np.mean(valid_loss), np.mean(valid_au_loss), np.mean(valid_ppg_loss), np.mean(valid_emotion_loss)

    def test(self, data_loader):
        # Print a message indicating the start of the testing process
        print("===Testing===")
        print('')

        # Check if the testing data is available and raise an error if not
        if data_loader["test"] is None:
            raise ValueError("No data for test")

        # Set the chunk length to the test chunk length specified in the configuration
        self.chunk_len = self.config.TEST.DATA.PREPROCESS.CHUNK_LENGTH

        # Initialize dictionaries to store predictions and labels for different tasks
        preds_dict_au = dict()
        labels_dict_au = dict()
        preds_dict_ppg = dict()
        labels_dict_ppg = dict()
        preds_dict_emotion = dict()
        labels_dict_emotion = dict()

        # IF ONLY_TEST MODE LOAD PRETRAINED MODEL
        if self.config.TOOLBOX_MODE == "only_test":
            # Load the pretrained model specified in the configuration
            model_path = self.config.INFERENCE.MODEL_PATH
            print("Testing uses pretrained model!")
            print('Model path:', model_path)

            # Check if the model path exists; raise an error if not
            if not os.path.exists(model_path):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")

        # IF USING MODEL FROM TRAINING
        else:
            # Load the model from a previous training session
            model_path = os.path.join(self.model_dir, self.model_file_name + '_Epoch' + str(self.used_epoch) + '.pth')
            print("Testing uses non-pretrained model!")
            print('Model path:', model_path)

            # Check if the model path exists and raise an error if not
            if not os.path.exists(model_path):
                raise ValueError("Something went wrong... cant find trained model...")
        
        print('')
            
        # Load the model's state dictionary and set it to evaluation mode
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.to(self.device)
        self.model.eval()

        # MODEL TESTING
        print("Running model evaluation on the testing dataset!")
        with torch.no_grad():
            # Iterate over the testing dataset
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):

                # Get the batch size from the test batch
                batch_size = test_batch[1].shape[0]

                # Format the data and labels, and send them to the appropriate device
                data, labels = test_batch[0], test_batch[1]
                data, labels = self.format_data_shape(data, labels)
                data, labels = self.send_data_to_device(data, labels)

                # Handling the weird bug where the final training batch to be of size 0
                if labels.shape[0] == 0:
                    continue

                # Get predictions from the model for AU, PPG tasks and Emotion task
                au_out, ppg_out, emotion_out = self.model(data)
                au_out = torch.sigmoid(au_out) 
                emotion_out = torch.softmax(emotion_out, dim=1)

                # Initialize flags and slice labels for each task
                if len(self.label_idx_test_au) > 0: # If test dataset has AU
                    labels_au = labels[:, self.label_idx_test_au]
                else: # If not set whole AU labels array to -1
                    labels_au = np.ones((batch_size, len(self.label_idx_train_au)))
                    labels_au = -1 * labels_au

                if len(self.label_idx_test_ppg) > 0: # if test dataset has PPG
                    labels_ppg = labels[:, self.label_idx_test_ppg]
                else: # if not set whole PPG labels array to -1
                    labels_ppg = np.ones((batch_size, len(self.label_idx_train_ppg)))
                    labels_ppg = -1 * labels_ppg

                if len(self.label_idx_test_emotion) > 0: # if test dataset has Emotion
                    labels_emotion = labels[:, self.label_idx_test_emotion]
                else: # if not set whole Emotion labels array to -1
                    labels_emotion = np.ones((batch_size, len(self.label_idx_train_emotion)))

                # Organize predictions and labels into dictionaries keyed by subject index
                for idx in range(batch_size):
                    # Skip if labels are cut off due to TSM data formatting
                    if idx * self.chunk_len >= labels.shape[0] and self.using_TSM:
                        continue 
                    
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])

                    # Add subject to prediction / label arrays if not already present
                    if subj_index not in preds_dict_ppg.keys():
                        preds_dict_au[subj_index] = dict()
                        labels_dict_au[subj_index] = dict()
                        preds_dict_ppg[subj_index] = dict()
                        labels_dict_ppg[subj_index] = dict()
                        preds_dict_emotion[subj_index] = dict()
                        labels_dict_emotion[subj_index] = dict()

                    # Append predictions and labels to subject dict
                    preds_dict_au[subj_index][sort_index] = au_out[idx * self.chunk_len:(idx + 1) * self.chunk_len] 
                    labels_dict_au[subj_index][sort_index] = labels_au[idx * self.chunk_len:(idx + 1) * self.chunk_len] 
                    preds_dict_ppg[subj_index][sort_index] = ppg_out[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    labels_dict_ppg[subj_index][sort_index] = labels_ppg[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    preds_dict_emotion[subj_index][sort_index] = emotion_out[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    labels_dict_emotion[subj_index][sort_index] = labels_emotion[idx * self.chunk_len:(idx + 1) * self.chunk_len]

        # Calculate evaluation metrics for each task using the predictions and labels
        # au_metric_dict = calculate_openface_au_metrics(preds_dict_au, labels_dict_au, self.config, self.au_label_names)
        # bvp_metric_dict = calculate_ppg_metrics(preds_dict_ppg, labels_dict_ppg, self.config)
        # emotion_metric_dict = calculate_emotion_metrics(preds_dict_emotion, labels_dict_emotion, self.config)