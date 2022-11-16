import functions
import visual_functions
from DataSaverClass import *
from HopfieldNetworkClass import *
from SystemCreatorClass import *
from ExperimentClass import *
from AnalyserClass import *

def main():

    single_analysis()

def photo_system_example():
    
    system = PhotoSystemCreator("flower2.jpg", 40, 'corner')
    network = HopfieldNetworkAsync(system.patterns)
    saver = DataSaverAsync(system.dimensions)
    network.dynamics(system.perturbed_pattern, saver, 5000000, 10000, 1000)
    saver.show_video()
    functions.verify_convergence(system.base_pattern, saver.states[-1], True)
    list_pattern = [system.reshape_pattern(system.perturbed_pattern), system.reshape_pattern(saver.states[5]), system.reshape_pattern(saver.states[20]), system.reshape_pattern(system.base_pattern)]
    list_titles = ['perturbed pattern', '5000 asynchronous updates', '20000 asynchronous updates', 'base pattern retrieved']
    visual_functions.visualize_patterns(list_pattern, list_titles)

def single_experiment():

    experiment = ExperimentCapacity(63, 'hebbian', 100)
    experiment.run()
    experiment.save()
    analyser = experiment.analyse()
    analyser.plot_curve()

def cumulated_general_curves():

    analyser_hebbian = GeneralCapacityAnalyser('hebbian')
    analyser_storkey = GeneralCapacityAnalyser('storkey')
        
    plt.plot(analyser_hebbian.sizes, analyser_hebbian.capacities, color = 'maroon', marker = '.', label = 'hebbian weights')
    plt.plot(analyser_storkey.sizes, analyser_storkey.capacities, color = 'olivedrab', marker = '.', label = 'storkey weights')
    plt.plot(analyser_hebbian.sizes, analyser_hebbian.theoretical_capacities, 'r-.', label = 'theoretical capacity hebbian')
    plt.plot(analyser_storkey.sizes, analyser_storkey.theoretical_capacities, 'g-.', label = 'theoretical capacity storkey')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("network size")
    plt.ylabel("maximal number of patterns for at least 90% retrieval")
    plt.title('Capacity over network size (logarithmic scale)')
    plt.show()

def run_experiments():

    sizes = [10, 18, 34, 63, 116, 215, 397, 733, 1354, 2500]
    
    for size in sizes:
        experiment = ExperimentRobustness(size, 'hebbian', 100)
        experiment.run()
        experiment.save()
        analyser = experiment.analyse()
        analyser.plot_curve()
    
def single_analysis():

    analyser = Analyser('HC10.h5')
    analyser.plot_curve()

if __name__ == '__main__':
    main()
