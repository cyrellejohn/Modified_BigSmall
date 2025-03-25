import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from scipy.stats import gaussian_kde

class BlandAltman():
    def __init__(self, gold_std, new_measure, config, averaged=False):
        """
        Initialize the BlandAltman class with the given measurements and configuration.

        Parameters:
        - gold_std: list, numpy array, or pandas series
            The gold standard measurements.
        - new_measure: list, numpy array, or pandas series
            The new measurements to compare against the gold standard.
        - config: object
            Configuration object containing settings for the toolbox, including paths for saving plots.
        - averaged: bool, optional
            Set to True if multiple observations from each participant are averaged together.

        This method performs the following tasks:
        - Converts input lists or numpy arrays to pandas series.
        - Calculates Bland-Altman statistics including mean error, standard deviation, and correlation.
        - Computes 95% confidence intervals for the mean error.
        - Configures the save path for plots based on the toolbox mode.
        - Creates the save path directory if it does not exist.
        """

        # Check that inputs are list or pandas series, convert to series if list
        if isinstance(gold_std,list) or isinstance(gold_std, (np.ndarray, np.generic) ):
            df = pd.DataFrame() # convert to pandas series
            df['gold_std'] = gold_std
            gold_std = df.gold_std

        elif not isinstance(gold_std,pd.Series):
            print('Error: Data type of gold_std is not a list or a Pandas series or Numpy array')

        if isinstance(new_measure,list) or isinstance(new_measure, (np.ndarray, np.generic) ):
            df2 = pd.DataFrame() # convert to pandas series
            df2['new_measure'] = new_measure
            new_measure = df2.new_measure

        elif not isinstance(new_measure,pd.Series):
            print('Error: Data type of new_measure is not a list or a Pandas series or Numpy array')

        self.gold_std = gold_std
        self.new_measure = new_measure

        # Calculate Bland-Altman statistics
        diffs = gold_std - new_measure
        self.mean_error = diffs.mean()
        self.std_error = diffs.std()
        self.mean_absolute_error = diffs.abs().mean()
        self.mean_squared_error = (diffs ** 2).mean()
        self.root_mean_squared_error = np.sqrt((diffs**2).mean())
        r = np.corrcoef(self.gold_std,self.new_measure)
        self.correlation = r[0,1] # correlation coefficient
        diffs_std = diffs.std() # 95% Confidence Intervals
        corr_std = np.sqrt(2*(diffs_std**2)) # if observations are averaged, use corrected standard deviation
        sqrt_sample_size = math.sqrt(self.gold_std.shape[0])

        if averaged:
            self.CI95 = [self.mean_error + 1.96 * corr_std , self.mean_error - 1.96 * corr_std]
        else:
            self.CI95 = [self.mean_error + 1.96 * diffs_std, self.mean_error - 1.96 * diffs_std]

        # Define save path
        if config.TOOLBOX_MODE == 'train_and_test' or config.TOOLBOX_MODE == 'only_test':
            self.save_path  = os.path.join(config.LOG.PATH, config.TEST.DATA.EXP_DATA_NAME, 'bland_altman_plots')
        elif config.TOOLBOX_MODE == 'unsupervised_method':
            self.save_path  = os.path.join(config.LOG.PATH, config.UNSUPERVISED.DATA.EXP_DATA_NAME, 'bland_altman_plots')
        else:
            raise ValueError('TOOLBOX_MODE only supports train_and_test, only_test, or unsupervised_method!')
        
        # Make the save path, if needed
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)

    def print_stats(self, round_amount = 5):
        """
        Prints the statistical measures calculated for the Bland-Altman analysis.

        Parameters:
        round_amount (int): The number of decimal places to round the statistics to. Default is 5.

        Outputs:
        Prints the following statistics:
        - Mean error: The average of the differences between the gold standard and the new measure.
        - Mean absolute error: The average of the absolute differences.
        - Mean squared error: The average of the squares of the differences.
        - Root mean squared error: The square root of the mean squared error.
        - Standard deviation error: The standard deviation of the differences.
        - Correlation: The correlation coefficient between the gold standard and the new measure.
        - +95% Confidence Interval: The upper bound of the 95% confidence interval for the mean error.
        - -95% Confidence Interval: The lower bound of the 95% confidence interval for the mean error.
        """

        print("Mean error = {}".format(round(self.mean_error,round_amount)))
        print("Mean absolute error = {}".format(round(self.mean_absolute_error,round_amount)))
        print("Mean squared error = {}".format(round(self.mean_squared_error,round_amount)))
        print("Root mean squared error = {}".format(round(self.root_mean_squared_error,round_amount)))
        print("Standard deviation error = {}".format(round(self.std_error,round_amount)))
        print("Correlation = {}".format(round(self.correlation,round_amount)))
        print("+95% Confidence Interval = {}".format(round(self.CI95[0],round_amount)))
        print("-95% Confidence Interval = {}".format(round(self.CI95[1],round_amount)))

    def return_stats(self):
        """
        Returns a dictionary containing various statistical measures
        calculated from the comparison between the gold standard and
        the new measurement.

        The dictionary includes:
        - mean_error: The average difference between the gold standard and new measurement.
        - mean_absolute_error: The average of the absolute differences.
        - mean_squared_error: The average of the squared differences.
        - root_mean_squared_error: The square root of the mean squared error.
        - correlation: The correlation coefficient between the two measurements.
        - CI_95%+: The upper bound of the 95% confidence interval for the mean error.
        - CI_95%-: The lower bound of the 95% confidence interval for the mean error.

        Returns:
            dict: A dictionary with keys as the names of the statistics and values as their respective calculated values.
        """
        
        stats_dict = {'mean_error': self.mean_error,
                      'mean_absolute_error': self.mean_absolute_error,
                      'mean_squared_error': self.mean_squared_error,
                      'root_mean_squared_error': self.root_mean_squared_error,
                      'correlation': self.correlation,
                      'CI_95%+': self.CI95[0],
                      'CI_95%-': self.CI95[1]}

        return stats_dict

    def rand_jitter(self, arr):
        """
        Adds a small amount of random noise (jitter) to an array of values.
        
        This function is useful for visualizing data points in a scatter plot
        where points might overlap, making it difficult to distinguish individual
        observations. The jitter is proportional to the range of the data.
        
        Parameters:
        arr (array-like): The input array of values to which jitter will be added.
        
        Returns:
        array-like: A new array with jitter added to each element.
        """
        # Calculate a small standard deviation based on 1% of the data range
        stdev = .01 * (max(arr) - min(arr))

        # Add random noise to each element in the array
        return arr + np.random.randn(len(arr)) * stdev

    def scatter_plot(self, x_label='Gold Standard', y_label='New Measure',
                    figure_size=(4,4), show_legend=True, the_title=' ',
                    file_name='BlandAltman_ScatterPlot.pdf', is_journal=False, 
                    measure_lower_lim=40, measure_upper_lim=150):
        """
        Creates and saves a scatter plot comparing the gold standard and new measure.

        Parameters:
        - x_label (str): Label for the x-axis. Default is 'Gold Standard'.
        - y_label (str): Label for the y-axis. Default is 'New Measure'.
        - figure_size (tuple): Size of the figure. Default is (4,4).
        - show_legend (bool): Whether to display the legend. Default is True.
        - the_title (str): Title of the plot. Default is an empty string.
        - file_name (str): Name of the file to save the plot. Default is 'BlandAltman_ScatterPlot.pdf'.
        - is_journal (bool): If True, adjusts font settings for journal publication. Default is False.
        - measure_lower_lim (int): Lower limit for the x and y axes. Default is 40.
        - measure_upper_lim (int): Upper limit for the x and y axes. Default is 150.

        Returns:
        - None: The plot is saved to the specified file path.
        """

        # Adjust font settings for journal publication if required
        if is_journal:
            import matplotlib
            matplotlib.rcParams['pdf.fonttype'] = 42
            matplotlib.rcParams['ps.fonttype'] = 42

        # Add jitter to the data to prevent overlap
        self.gold_std = self.rand_jitter(self.gold_std)
        self.new_measure = self.rand_jitter(self.new_measure)

        # Create the figure and axis
        fig = plt.figure(figsize=figure_size)
        ax=fig.add_axes([0,0,1,1])

        # Calculate the density of points for coloring
        xy = np.vstack([self.gold_std,self.new_measure])
        z = gaussian_kde(xy)(xy)

        # Plot the scatter plot with density-based coloring
        ax.scatter(self.gold_std,self.new_measure, c=z, s=50)

        # Plot a line of equality (slope = 1)
        x_vals = np.array(ax.get_xlim())
        ax.plot(x_vals,x_vals,'--',color='black', label='Line of Slope = 1')

        # Set labels, title, and grid
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(the_title)
        ax.grid()

        # Set axis limits
        plt.xlim(measure_lower_lim, measure_upper_lim)
        plt.ylim(measure_lower_lim, measure_upper_lim)

        # Save the plot to the specified file path
        plt.savefig(os.path.join(self.save_path, file_name),bbox_inches='tight', dpi=300)
        print(f"Saved {file_name} to {self.save_path}.")

    def difference_plot(self, x_label='Difference between rPPG HR and ECG HR [bpm]',
                        y_label='Average of rPPG HR and ECG HR [bpm]', averaged=False,
                        figure_size=(4,4), show_legend=True, the_title='', 
                        file_name='BlandAltman_DifferencePlot.pdf', is_journal=False):
        """
        Creates a Bland-Altman difference plot to analyze the agreement between two measurement methods.

        Parameters:
        - x_label (str): Label for the x-axis. Default is 'Difference between rPPG HR and ECG HR [bpm]'.
        - y_label (str): Label for the y-axis. Default is 'Average of rPPG HR and ECG HR [bpm]'.
        - averaged (bool): Flag indicating if the data is averaged. Default is False.
        - figure_size (tuple): Size of the plot in inches. Default is (4, 4).
        - show_legend (bool): Whether to display the legend. Default is True.
        - the_title (str): Title of the plot. Default is an empty string.
        - file_name (str): Name of the file to save the plot. Default is 'BlandAltman_DifferencePlot.pdf'.
        - is_journal (bool): If True, adjusts font settings for journal publication. Default is False.

        The method calculates the differences and averages of the gold standard and new measurements,
        then plots these values. It also includes lines for the mean error and 95% confidence intervals.
        The plot is saved to the specified file path.
        """

        # Adjust font settings for journal publication if needed
        if is_journal:
            matplotlib.rcParams['pdf.fonttype'] = 42
            matplotlib.rcParams['ps.fonttype'] = 42

        # Calculate differences and averages
        diffs = self.gold_std - self.new_measure
        avgs = (self.gold_std + self.new_measure) / 2

        # Create the plot
        fig = plt.figure(figsize=figure_size)
        ax = fig.add_axes([0,0,1,1])
        xy = np.vstack([avgs,diffs])
        z = gaussian_kde(xy)(xy)
        ax.scatter(avgs,diffs, c=z, label='Observations')

        # Add horizontal lines for mean error and confidence intervals
        x_vals = np.array(ax.get_xlim())
        ax.axhline(self.mean_error,color='black',label='Mean Error')
        ax.axhline(self.CI95[0],color='black',linestyle='--',label='+95% Confidence Interval')
        ax.axhline(self.CI95[1],color='black',linestyle='--',label='-95% Confidence Interval')

        # Set labels, title, and grid
        ax.set_ylabel(x_label)
        ax.set_xlabel(y_label)
        ax.set_title(the_title)
        if show_legend:
            ax.legend()
        ax.grid()

        # Save the plot
        plt.savefig(os.path.join(self.save_path, file_name),bbox_inches='tight', dpi=100)
        print(f"Saved {file_name} to {self.save_path}.")