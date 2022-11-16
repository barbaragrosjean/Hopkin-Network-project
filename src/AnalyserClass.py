import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

'''
This file contains classes allowing to analyse experimental data :
    - Analyser
        - CapacityAnalyser(Anlayser)
        - RobustnessAnalyser(Analyser)
        - GeneralAnalyser
            - GeneralCapacityAnalyser
            - GeneralRobustnessAnalyser
'''

class Analyser :
    '''
    This class allows to analyse the data of an experiment. The model tested is a network with a given size. The aims of this class is to plot the model feature tested.
    
    this class contains the following methods:
    - __init__
    - results_from_experiment
    - results_from_file
    - create_file_name
    - plot curve (abstract)
    - save_plot
    - collect_data_files
    
    and has the following attributes:
    - exp_result
    '''
    def __new__(cls, exp_results):
        '''This allows to define an analyser simply by calling it with the superclass. This method will recognize which experiment type is performed an use the corresponding subclass.'''

        if isinstance(exp_results, str): 
            if exp_results[1] == 'C': return super().__new__(CapacityAnalyser)
            if exp_results[1] == 'R': return super().__new__(RobustnessAnalyser)

        return super().__new__(cls)

    def __init__(self, exp_result) :
        '''
        This methode initializes the data given by an experiment either by searching in the corresponding file or by directly using the results of an experiment.
    
        Parameters
        ----------
        exp_result: array of dictionaries or string
            list containing the results of an experiment or string corresponding to the name of the file containing the results
        '''
        if isinstance(exp_result, str): self.exp_result = self.results_from_file(exp_result)
        else: self.exp_result = self.results_from_experiment(exp_result)

    def results_from_experiment(self, results):
        '''Transforms the array of dictionaries given by the experiment into a dictionary of lists, which is more handy for plotting'''

        return {key: [dic[key] for dic in results] for key in results[0]}

    def results_from_file(self, filename):
        '''Creates a dictionary of lists from a .h5 file'''
        
        filepath = os.path.join(os.path.dirname(__file__), os.pardir)
        filepath = os.path.join(filepath, 'res', filename)

        if not os.path.exists(filepath):
            raise ValueError("The entered experiment data file does not exist.")
               
        df = pd.read_hdf(str(filepath))

        return {key: list(df.loc[:, key]) for key in list(df)} #Note that list(df) is the list of all column headers (size, weight_rule, ...)

    def create_file_name(self, exp_code):
        '''Creates the file name corresponding to the experiment that is analysed'''

        if self.exp_result["weight_rule"][0] == 'hebbian': weight_code = 'H'
        elif self.exp_result["weight_rule"][0] == 'storkey': weight_code = 'S'
    
        filename = weight_code + exp_code + str(self.exp_result["network_size"][0])
        return filename

    def plot_curve(self, save = False):
        raise NotImplementedError

    def get_plot_values(self):
        '''
        Returns values needed for experiment plotting, such as the percentage of retrieved patterns or the line colour.

        Returns
        -------
        x_values: list
            x values for the plot, depending on the type of experiment
        y_values: list
            Percentage of how many trials have led to pattern retrieval
        x_label: string
            Label used for the x axis of the plot
        y_label: string
            Label used for the y axis of the plot
        color: string
            Color used for all subplots.
        theoretical_capacity: float
            Theoretical value of the asymptotic capacity for a network of a certain size
        '''

        raise NotImplementedError

    @staticmethod
    def save_plot(filename):
        '''
        This method allows to save a given figure with the name given in parameter.
    
        Parameters
        ----------
        filename : string
            Name the figure will be saved with.
        '''

        path = os.path.join(os.path.dirname(__file__), os.pardir)  #parent directory of src
        path = os.path.join(path, 'plot')                          #going into directory plot
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename + '.png')
        plt.savefig(path)

        print("Figure {} saved under: ".format(filename), path, "\n")

    @staticmethod
    def collect_data_files(weight_code, exp_code):
        '''Static method collecting all the data file names corresponding to certain experiments in increasing network size order'''
        
        path = os.path.join(os.path.dirname(__file__), os.pardir)
        res_path = os.path.join(path, 'res')
        all_files = os.listdir(res_path)
        experiment_files = list()
        sizes = list()

        for file in all_files:
            if file[0] == weight_code and file[1] == exp_code:
                experiment_files.append(file)
                sizes.append(int(file[2:-3]))

        sorted_indices = np.array(sizes).argsort()
        experiment_files = np.array(experiment_files)[sorted_indices]

        return experiment_files
        
    @staticmethod
    def subplot_experiments(data_files, dimensions, filename):
        '''
        Makes one plot using the data from the list of files in parameter.

        Parameters
        ----------
        data_files: list or array or strings
            names of all the files that will be included in the plot
        dimensions: tuple of integers
            gives how many subplots there will be per line and how many lines are needed
        filename: string
            name the file will be saved with in folder plot
        '''

        if(dimensions[0] * dimensions[1] != len(data_files)): raise ValueError("The number of experiments does not match the number of subplots")
       
        weight_code, exp_code = data_files[0][0], data_files[0][1]
        fig, axs = plt.subplots(dimensions[0], dimensions[1], figsize = (4 * dimensions[1], 2.75 * dimensions[0]), sharey = True)
        plt.tight_layout(pad = 2.5)
        plt.subplots_adjust(top = 0.92)

        if exp_code == 'C': title = 'Capacity experiments '
        if exp_code == 'R': title = 'Robustness experiments '

        if weight_code == 'H': title += 'using Hebbian weights'
        if weight_code == 'S': title += 'using Storkey weights'

        fig.suptitle(title, fontsize = 16, y = 0.97)

        for x in range(dimensions[0]):
            for y in range(dimensions[1]):
                if exp_code == 'C': analyser = CapacityAnalyser(data_files[y + dimensions[1] * x])
                if exp_code == 'R': analyser = RobustnessAnalyser(data_files[y + dimensions[1] * x])

                x_values, y_values, x_label, y_label, color, theoretical_capacity = analyser.get_plot_values()
                axs[x][y].plot(x_values, y_values, color = color, marker = '.')
                axs[x][y].set_title('Network size = ' + str(analyser.exp_result["network_size"][0]), fontsize = 14)
                if y == 0: axs[x][y].set(ylabel = y_label)
                if x == dimensions[0] - 1: axs[x][y].set(xlabel = x_label)

                if exp_code == 'C': axs[x][y].axvline(theoretical_capacity)

        Analyser.save_plot(filename)

class CapacityAnalyser(Analyser):
    '''This class allows to analyse data from a capacity experiment.'''

    def get_plot_values(self):
        '''Returns values needed specifically for capacity plotting, such as the percentage of retrieved patterns or the line colour.'''

        x_values = self.exp_result["num_patterns"]
        y_values = [element * 100 for element in self.exp_result["match_frac"]]

        x_label = "number of patterns of the network"
        y_label = "% retrieval"

        #Choosing colour depending on the weight rule used
        if str(self.exp_result["weight_rule"][0]) == 'hebbian': color = 'black'
        else: color ='m'

        theoretical_capacity = self.exp_result["theoretical_capacity"][0]

        return x_values, y_values, x_label, y_label, color, theoretical_capacity

    def plot_curve(self, save = False):
        '''This method allows to plot the fraction of experiment passes (with a network still convergent) over the studied feature (capacity). This method saves the plot.'''
        
        x_values, y_values, x_label, y_label, color, theoretical_capacity = self.get_plot_values()
        title = 'Capacity curve using ' + str(self.exp_result["weight_rule"][0]).capitalize() + ' weights for network size = ' + str(self.exp_result["network_size"][0])

        plt.plot(x_values, y_values, color = color, marker = '.', label = "experimental data")
        plt.axvline(theoretical_capacity, label = 'theoretical capacity')
        plt.legend()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        if save: Analyser.save_plot(self.create_file_name('C'))
        plt.show()

class RobustnessAnalyser(Analyser):
    '''This class permits analysing data from a robustness experiment.'''

    def get_plot_values(self):
        '''Returns values needed specifically for plotting, such as the network size or the percentage of retrieved patterns.'''

        x_values = [element * (100/self.exp_result["network_size"][0]) for element in self.exp_result["num_perturb"]] #Normalising the data
        y_values = [element * 100 for element in self.exp_result["match_frac"]]

        x_label = "% perturbation"
        y_label = "% retrieval"

        #Choosing colour depending on the weight rule used
        if str(self.exp_result["weight_rule"][0]) == 'hebbian': color = 'darkgreen'
        else: color ='darkorange'

        return x_values, y_values, x_label, y_label, color, 0 #Returns a zero in order to have the same number of returns than the analogous capacity method

    def plot_curve(self, save = False):
        '''This method allows to plot the fraction of experiment passes (with a network still convergent) over the studied feature (robustness). This method saves the plot.'''
        
        x_values, y_values, x_label, y_label, color, _ = self.get_plot_values()
        title = 'Robustness curve using ' + str(self.exp_result["weight_rule"][0]).capitalize() + ' weights for network size = ' + str(self.exp_result["network_size"][0])
            
        plt.plot(x_values, y_values, color = color, marker = '.', label = 'experimental_data')
        plt.legend()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        if save: Analyser.save_plot(self.create_file_name('R'))
        plt.show()

class GeneralAnalyser(Analyser):
    '''
    This class allow to analyse the capacity for different sizes of network and contains the following methods:
    - __init__
    - fetch data
    - plot_curve
    '''

    def __init__(self, weight_rule):
        '''
        Initializer of class GeneralAnalyser.
            
        Parameters
        ----------
        weight rule: string
            Either 'hebbian' or 'storkey', depending on which rule's capacity we want to analyse
        '''

        #Setting the code of the rule we want to analyse data of
        if weight_rule == 'hebbian': self.weight_code, self.rule_name = 'H', 'Hebbian weights'
        elif weight_rule == 'storkey': self.weight_code, self.rule_name = 'S', 'Storkey weights'
        else: raise ValueError("The entered weight rule is not defined. Use 'storkey' or 'hebbian'")

        #Loading all the files in the directory in which data files are stored to find all relevant files
        path = os.path.join(os.path.dirname(__file__), os.pardir)
        self.res_path = os.path.join(path, 'res')

        self.fetch_data()
        
    def fetch_data(self):
        raise NotImplementedError
    
    def plot_curve(self):
        raise NotImplementedError

class GeneralCapacityAnalyser(GeneralAnalyser):
    '''
    Subclass of GeneralAnalyser specific to capacity experiments. Defines some abstract methods of the superclass and adds following methods:
    - find_capacity

    and attributes:
    - sizes: network sizes studied
    - capacitites: network capacities corresponding to the different network sizes
    - theoretical_capacities: asymptotic capacity for this network size
    '''

    def fetch_data(self):
        '''This method looks through the res folder for all the data files that correspond to the experiment type we analyse (with the right weights)'''
        
        #Initialising empty lists and sets used for plotting
        sizes, capacities, theoretical_capacities = list(), list(), list()
        
        files_list = os.listdir(self.res_path)
        for file in files_list:

            #Using only data files which correspond to the experiments that we want to analyse
            if file[0] == self.weight_code and file[1] == 'C':
                
                filepath = os.path.join(self.res_path, file)
                df = pd.read_hdf(str(filepath))

                capacities.append(self.find_capacity(df))
                sizes.append(df.loc[0,"network_size"])
                theoretical_capacities.append(df.loc[0, "theoretical_capacity"])

        #Initialising sorted list as class attributes
        sorted_indices = np.array(sizes).argsort()
        self.sizes = np.array(sizes)[sorted_indices]
        self.capacities = np.array(capacities)[sorted_indices]
        self.theoretical_capacities = np.array(theoretical_capacities)[sorted_indices]

    def find_capacity(self, dataframe):
        '''
        This method allows to find the capacity of a model for a given network size and a set of trials with different number of patterns memorised.
        
        Returns
        -------
        capacity : int
            the number of memorized pattern sush that we pass 90% of the trials.
        '''

        capacity = 0

        for i in range(dataframe.shape[0]):
            if dataframe.loc[i, "match_frac"] >= 0.9 :
                    capacity = dataframe.loc[i, "num_patterns"]
               
        return capacity
     
    def plot_curve(self, save = False, log = True):
        '''This method plots the model capacity vs the size of network and saves the plot.'''

        #Choosing colour depending on the weight rule used
        if self.weight_code == 'H': color = 'maroon'
        else: color ='olivedrab'

        title = 'Capacity over network size using ' + self.rule_name
              
        #plot the capacities vs the size of network
        plt.plot(self.sizes, self.capacities, color = color, marker = '.', label = 'experimental data')
        plt.plot(self.sizes, self.theoretical_capacities, 'b-.', label = 'theoretical capacity')
        plt.legend()
        plt.xlabel("network size")
        plt.ylabel("maximal number of patterns for at least 90% retrieval")

        if log:
            title += ' (logarithmic scale)'
            plt.xscale('log')
            plt.yscale('log')

        plt.title(title)
        if save: Analyser.save_plot('GeneralCapacity')
        plt.show()
        
class GeneralRobustnessAnalyser(GeneralAnalyser):  
    '''
    Subclass of GeneralAnalyser specific to robustness experiments. Defines some abstract methods of the superclass and adds following methods:
    - find_capacity

    and attributes:
    - sizes: network sizes studied
    - robustness: network's for a certain network size
    '''

    def fetch_data(self):
        '''This method looks through the res folder for all the data files that correspond to the experiment type we analyse (with the right weights)'''
        
        #Initialising empty lists and sets used for plotting
        sizes, robustness = list(), list()
        
        files_list = os.listdir(self.res_path)
        for file in files_list:

            #Using only data files which correspond to the experiments that we want to analyse
            if file[0] == self.weight_code and file[1] == 'R':
                
                filepath = os.path.join(self.res_path, file)
                df = pd.read_hdf(str(filepath))

                robustness.append(self.find_robustness(df))
                sizes.append(df.loc[0,"network_size"])

        #Initialising sorted list as class attributes
        sorted_indices = np.array(sizes).argsort()
        self.sizes = np.array(sizes)[sorted_indices]
        self.robustness = 100 * np.array(robustness)[sorted_indices] / self.sizes

    def find_robustness(self, dataframe):
        '''
        This method allows to find the robustness of a model for a given network size and a set of trials with different percentages of perturbation.
        
        Returns
        -------
        robustness : int
            the percentage of perturbation such that 80% of the trials are passed.
        '''

        robustness = 0

        for i in range(dataframe.shape[0]):
            if dataframe.loc[i, "match_frac"] >= 0.8 :
                    robustness = dataframe.loc[i, "num_perturb"]
               
        return robustness
     
    def plot_curve(self, save = False, log = True):
        '''This method plots the model's robustness vs the size of network and saves the plot.'''

        #Choosing colour depending on the weight rule used
        if self.weight_code == 'H': color = 'teal'
        else: color ='red'

        title = 'Robustness over network size using ' + self.rule_name
              
        #plot the robustness vs the size of network
        plt.plot(self.sizes, self.robustness, color = color, marker = '.', label = 'experimental robustness result')
        plt.legend()
        plt.xlabel("network size")
        plt.ylabel("maximal perturbation (%) for at least 80% retrieval")

        if log:
            plt.xscale('log')
            title += ' (logarithmic scale)'

        plt.title(title)
        if save: Analyser.save_plot('GeneralRobustness')
        plt.show()
