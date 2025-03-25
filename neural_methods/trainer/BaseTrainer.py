"""Base Trainer""" 
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator

class BaseTrainer:
    def __init__(self):
        pass
    
    @staticmethod
    def add_trainer_args(parser):
        """Adds arguments to Parser for training process"""
        parser.add_argument('--lr', default=None, type=float)
        parser.add_argument('--model_file_name', default=None, type=float)
        return parser

    def train(self, data_loader):
        pass

    def valid(self, data_loader):
        pass

    def test(self):
        pass

    def save_test_outputs(self, predictions, labels, config):
        """
        Saves the test outputs (predictions and labels) to a pickle file.

        Parameters:
        - predictions: The predicted values from the model.
        - labels: The true labels corresponding to the predictions.
        - config: Configuration object containing settings for output directory,
                  toolbox mode, and other relevant parameters.

        The method constructs a filename based on the toolbox mode and saves
        the predictions, labels, label type, and sampling frequency to a
        pickle file in the specified output directory.
        """

        # Retrieve the output directory from the configuration
        output_dir = config.TEST.OUTPUT_SAVE_DIR
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True) # Create directory if it doesn't exist
        
        # Determine the filename ID based on the toolbox mode
        if config.TOOLBOX_MODE == 'train_and_test':
            filename_id = self.model_file_name
        elif config.TOOLBOX_MODE == 'only_test':
            model_file_root = config.INFERENCE.MODEL_PATH.split("/")[-1].split(".pth")[0]
            filename_id = model_file_root + "_" + config.TEST.DATA.DATASET
        else:
            raise ValueError('Metrics.py evaluation only supports train_and_test and only_test!')
        
        # Construct the full path for the output file
        output_path = os.path.join(output_dir, filename_id + '_outputs.pickle')

        # Prepare the data dictionary to be saved
        data = dict()
        data['predictions'] = predictions
        data['labels'] = labels
        data['label_type'] = config.TEST.DATA.PREPROCESS.LABEL_TYPE
        data['fs'] = config.TEST.DATA.FS

        # Save the data dictionary to a pickle file
        with open(output_path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Print confirmation message
        print('Saving outputs to:', output_path)

    def plot_losses_and_lrs(self, train_loss, valid_loss, lrs, config):
        """
        Generates and saves plots for training and validation losses, as well as learning rates.

        Parameters:
        - train_loss: List of training loss values for each epoch.
        - valid_loss: List of validation loss values for each epoch.
        - lrs: List of learning rate values for each scheduler step.
        - config: Configuration object containing paths and settings.

        The method creates two plots:
        1. A plot of training and validation losses over epochs.
        2. A plot of learning rates over scheduler steps.

        The plots are saved as PDF files in a directory specified by the config.
        """

        # Construct the output directory path for saving plots
        output_dir = os.path.join(config.LOG.PATH, config.TRAIN.DATA.EXP_DATA_NAME, 'plots')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Determine the filename ID based on the toolbox mode
        if config.TOOLBOX_MODE == 'train_and_test':
            filename_id = self.model_file_name
        else:
            raise ValueError('Metrics.py evaluation only supports train_and_test and only_test!')
        
        # Create a plot for training and validation losses
        plt.figure(figsize=(10, 6))
        epochs = range(0, len(train_loss))  # Integer values for x-axis
        plt.plot(epochs, train_loss, label='Training Loss')
        if len(valid_loss) > 0:
            plt.plot(epochs, valid_loss, label='Validation Loss')
        else:
            print("The list of validation losses is empty. The validation loss will not be plotted!")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{filename_id} Losses')
        plt.legend()
        plt.xticks(epochs)

        # Set y-axis ticks with more granularity
        ax = plt.gca()
        ax.yaxis.set_major_locator(MaxNLocator(integer=False, prune='both'))

        # Save the loss plot as a PDF file
        loss_plot_filename = os.path.join(output_dir, filename_id + '_losses.pdf')
        plt.savefig(loss_plot_filename, dpi=300)
        plt.close()

        # Create a separate plot for learning rates
        plt.figure(figsize=(6, 4))
        scheduler_steps = range(0, len(lrs))
        plt.plot(scheduler_steps, lrs, label='Learning Rate')
        plt.xlabel('Scheduler Step')
        plt.ylabel('Learning Rate')
        plt.title(f'{filename_id} LR Schedule')
        plt.legend()

        # Set y-axis values in scientific notation
        ax = plt.gca()
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))  # Force scientific notation

        # Save the learning rate plot as a PDF file
        lr_plot_filename = os.path.join(output_dir, filename_id + '_learning_rates.pdf')
        plt.savefig(lr_plot_filename, bbox_inches='tight', dpi=300)
        plt.close()

        # Print a message indicating where the plots have been saved
        print('Saving plots of losses and learning rates to:', output_dir)