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
        Efficiently generates and saves loss and learning rate plots.
        """

        if config.TOOLBOX_MODE != 'train_and_test':
            raise ValueError("Metrics.py evaluation only supports 'train_and_test' and 'only_test'!")

        filename_id = self.model_file_name
        output_dir = os.path.join(config.LOG.PATH, config.TRAIN.DATA.EXP_DATA_NAME, 'plots')
        os.makedirs(output_dir, exist_ok=True)

        def _save_plot(fig_size, x, y_list, labels, title, xlabel, ylabel, path, y_format=None):
            plt.figure(figsize=fig_size)
            for y, label in zip(y_list, labels):
                plt.plot(x, y, label=label)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.legend()

            ax = plt.gca()
            if y_format == "int":
                ax.yaxis.set_major_locator(MaxNLocator(integer=False, prune="both"))
            elif y_format == "sci":
                ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            plt.tight_layout()
            plt.savefig(path, dpi=300)
            plt.close()

        epochs = list(range(len(train_loss)))
        loss_labels = ['Training Loss']
        loss_values = [train_loss]
        if valid_loss:
            loss_labels.append('Validation Loss')
            loss_values.append(valid_loss)
        else:
            print("Validation loss is empty; only training loss will be plotted.")

        _save_plot(fig_size=(10, 6),
                   x=epochs,
                   y_list=loss_values,
                   labels=loss_labels,
                   title=f'{filename_id} Losses',
                   xlabel='Epoch',
                   ylabel='Loss',
                   path=os.path.join(output_dir, f'{filename_id}_losses.pdf'),
                   y_format='int')

        _save_plot(fig_size=(6, 4),
                   x=list(range(len(lrs))),
                   y_list=[lrs],
                   labels=['Learning Rate'],
                   title=f'{filename_id} LR Schedule',
                   xlabel='Scheduler Step',
                   ylabel='Learning Rate',
                   path=os.path.join(output_dir, f'{filename_id}_learning_rates.pdf'),
                   y_format='sci')

        print(f"Saved loss and learning rate plots to: {output_dir}")