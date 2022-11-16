import numpy as np
import pandas as pd
from functions import verify_convergence
from DataSaverClass import *
from HopfieldNetworkClass import *
from SystemCreatorClass import *
from AnalyserClass import *

class Experiment:
    '''
    This class allows to run a set of experiments to test neurons network capacity and robustness and save the results as raw data and graphs.

    This class contains the following methods:
    - __init__
    - experiment
    - run
    - save

    And has as attributes
    - start_values: list of starting values for which an experiment is performed 
    - results: list of dictionnaries with the results for all sub-experiments performed
    '''

    def __init__(self, size, weight_rule, num_trial = 10, max_iter = 100):
        '''
        This method initializes the attributes of the Experiment class.

        Parameters
        ----------
        size : integer
            The size of the neurons networks.
        weight_rule : string
            Represents the chosen rule to calculate the weights of the given set of patterns.
        num_trial : integer
            The number of time for which we are going to do each experiments
        max_iter : integer
            The maximum number of steps allowed before it is considered convergence will not occur.
        '''

        self.capacity = HopfieldNetwork(SystemCreator.generate_patterns(1,size), weight_rule).capacity
        self.__size, self.__weight_rule, self.__num_trial, self.__max_iter = size, weight_rule, num_trial, max_iter

    def experiment(self):
        raise NotImplementedError

    def run(self):
        '''Calling this function runs the series of experiments for a certain size of neuronal network'''

        self.experiments = np.vectorize(self.experiment)
        self.results = self.experiments(self.start_values, self.__size, self.__weight_rule, self.__num_trial, self.__max_iter)
        
    def analyse(self):
        '''
        taking the result of the experiment and creat a instance of analyser.
        
        '''
        raise NotImplementedError
    
    def save(self):
        '''
        Method allowing to save the experimental results as a pandas dataframe and a matplotlib plot. The name of the output file will contain:
        - first letter: 'S' if experiment was run with Storkey rule, 'H' if run with Hebbian rule
        - second letter: 'C' for a capacity experiment, 'R' for a robustness experiment
        - following letters: size of the network worked with in the experiment
        '''

        if self.__weight_rule == 'hebbian': weight_code = 'H'
        elif self.__weight_rule == 'storkey': weight_code = 'S'
        filename = weight_code + self.experiment_code + str(self.__size)

        path = os.path.join(os.path.dirname(__file__), os.pardir)  #parent directory of src
        path = os.path.join(path, 'res')                           #going into new directory animations
        os.makedirs(path, exist_ok=True)                           #creating new directory animations if needed
        path = os.path.join(path, filename)

        df = pd.DataFrame(self.results.tolist())
        df.to_hdf(path + '.h5', key='df')
        
        print("File {} saved under: ".format(filename), path, "\n")
           
class ExperimentCapacity(Experiment):

    def __init__(self, size, weight_rule, num_trial = 10, max_iter = 100):
        '''Besides calling the super-class initialiser, experiment specific attributes are computed'''

        super().__init__(size, weight_rule, num_trial=num_trial, max_iter=max_iter)
        self.experiment_code = 'C'

        #10 experiments are performed starting with a different number of starting patterns
        self.start_values = np.unique(np.linspace(0.5 * self.capacity, 2 * self.capacity, 10).astype(int))

    def experiment(self, num_patterns, size, weight_rule, num_trial, max_iter):
        '''
        This method returns a dictionnary containing the results of performing a capacity experiment for a certain number of patterns.

        Returns
        -------
        dic_result: dictionary
            The results of an experiment
        '''
        num_perturb = int(size * 0.2)
        system = SystemCreator(int(num_patterns), int(size), num_perturb)
        network = HopfieldNetworkSync(system.patterns, weight_rule)
        saver = DataSaverSync()

        success = 0
        for trial in range(0, num_trial):
            system.next_base_pattern()
            network.dynamics(system.perturbed_pattern, saver, int(max_iter))
            if verify_convergence(system.base_pattern, saver.states[-1]): success += 1
            saver.reset()
            
        dic_result = {
            "network_size": size,
            "weight_rule": weight_rule, 
            "num_patterns": num_patterns, 
            "num_perturb": num_perturb, 
            "match_frac": success/num_trial, 
            "theoretical_capacity": self.capacity
        }

        return dic_result
    
    def analyse(self):
        return CapacityAnalyser(self.results)
      
class ExperimentRobustness(Experiment):

    def __init__(self, size, weight_rule, num_trial=10, max_iter=100):
        '''Besides calling the super-class initialiser, experiment specific attributes are computed'''

        super().__init__(size, weight_rule, num_trial=num_trial, max_iter=max_iter)
        self.experiment_code = 'R'

        #Only one network and system are needed for this experiment
        self.__num_patterns = int(0.75 * self.capacity)
        self.system = SystemCreator(self.__num_patterns, int(size), 0)
        self.network = HopfieldNetworkSync(self.system.patterns, weight_rule)

        self.start_values = np.unique((np.arange(0.2, 1.05, 0.05) * size).astype(int))

    def experiment(self, num_perturb, size, weight_rule, num_trial, max_iter):
        '''
        This method gives the result of a robustness experiment for a certain number of perturbations.

        Returns
        -------
        dic_result: dictionary
            The results of an experiment
        '''

        saver = DataSaverSync()

        success = 0
        for trial in range(0, num_trial):
            self.system.change_num_perturb(int(num_perturb))
            self.network.dynamics(self.system.perturbed_pattern, saver, int(max_iter))
            if verify_convergence(self.system.base_pattern, saver.states[-1]): success += 1
            saver.reset()
            
        dic_result = {
            "network_size": size, 
            "weight_rule": weight_rule, 
            "num_patterns": self.__num_patterns, 
            "num_perturb": num_perturb, 
            "match_frac": success/num_trial
        }

        return dic_result

    def analyse(self):
        return RobustnessAnalyser(self.results)