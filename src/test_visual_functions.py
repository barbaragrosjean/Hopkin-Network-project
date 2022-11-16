import visual_functions
import pytest
import numpy as np
from DataSaverClass import *
from HopfieldNetworkClass import *
from SystemCreatorClass import *

'''Overview of functions contained in this module:
     - test_reshape_patterns_sync
     - test_reshape_patterns_sync
     - test_visualize_pattern
     - test_video_sync
     - test_video_async
     - test_energy_decreasing_sync
     - test_energy_decreasing_async
     - test_doctest_visual_functions
'''

def test_reshape_states_visualization_sync():
    '''
    This function allows to test if the reshaping of a set of patterns to square matrices happens as it should (mainly testing dimensions).
    '''
    #Creating a system and running and saving the dynamic network

    system = SystemCreator(50, 2500, 1000, True)
    data = DataSaverSync(system.dimensions)
    network_sync = HopfieldNetworkSync(system.patterns)
    network_sync.dynamics(system.base_pattern, data)
    reshaped = data.reshape_states()

    #Testing if the given matrix flattened again is still equal to the inital pattern
    height, width = data.dimensions
    assert (width, height) == reshaped[0].shape
    assert len(data.states[0]) == width * height
    assert np.all(data.states[0] == np.ravel(reshaped[0]))

    #Testing if raising exceptions works
    wrong_data = DataSaverSync((2,2))
    wrong_data.states.append(np.array([1,-1,1])) #Appending a state whose size does not fit the dimensions of the saver
    with pytest.raises(ValueError):
           wrong_data.reshape_states()
     
def test_reshape_states_visualization_async():
    '''
    This function allows to test if the reshaping of a set of patterns to square matrices happens as it should (mainly testing dimensions).
    '''
    #Creating a system and running and saving the dynamic network

    system = SystemCreator(50, 2500, 1000, True)
    data = DataSaverAsync(system.dimensions)
    network_async = HopfieldNetworkAsync(system.patterns)
    network_async.dynamics(system.base_pattern, data)
    reshaped = data.reshape_states()
     
    #Testing dimensions and equality after reshaping
    height, width = data.dimensions
    assert (width, height) == reshaped[0].shape
    assert len(data.states[0]) == width * height
    assert np.all(np.ravel(reshaped[0]) == data.states[0])
    
    #Testing if raising exceptions works
    wrong_data = DataSaverSync((2,2))
    wrong_data.states.append(np.array([1,-1,1])) #Appending a state whose size does not match with the dimensions of the saver
    with pytest.raises(ValueError):
           wrong_data.reshape_states()
       
def test_visualize_pattern():
   
    '''
    This function allows to test the function visualize_pattern, which allows to display a visual representation of a pattern as a plot. This test simply consists in plotting a checkerboard.
    '''

    pattern = visual_functions.generate_checkerboard(144,4)
    visual_functions.visualize_pattern(pattern, 'Checkerboard of size 144 and submatrix size 4')
 
def test_video_sync():
    
    '''
    This function allows to visualize the convergence of a perturbed checkerboard pattern with a synchronous update. It therefore displays and saves videos which contain a visualisation of each n-th step taken until the convergence.
    '''
    #Creating a system and running and saving the dynamic network
    system = SystemCreator(15, 144, 30, True)
    network = HopfieldNetworkSync(system.patterns)
    data = DataSaverSync(system.dimensions)
    network.dynamics(system.perturbed_pattern, data)
    
    #show and save the video showing the evolution of the system
    data.show_video()
    data.save_video('Synchronous')
    
def test_video_async():
    '''
    This function allows to visualize the convergence of a perturbed checkerboard pattern with a asynchronous update. It therefore displays and saves videos which contain a visualisation of each n-th step taken until the convergence.
    '''
    #Creating a system and running and saving the dynamic network
    system = SystemCreator(15, 144, 30, True)
    network = HopfieldNetworkAsync(system.patterns)
    saver = DataSaverAsync(system.dimensions)
    network.dynamics(system.perturbed_pattern, saver)
    
    #show and save the video showing the evolution of the system
    saver.show_video()
    print(len(saver.states))
    saver.save_video('Asynchronous')
    
def test_energy_descreasing_sync():

    '''
    This function tests if the energy function is decreasing. It may be modified in order to test all states.

    Notes
    -----
    - The function first tests the energy function by comparing values and then shows a graph of the energy states throughout updating.
    '''

    #Creating a system and running and saving the dynamic network
    system = SystemCreator(50, 2500, 1000)
    data = DataSaverSync()
    network_sync = HopfieldNetworkSync(system.patterns)
    network_sync.dynamics(system.perturbed_pattern, data)

    #control if the list energy is decreasing
    assert all(data.energies[i] >= data.energies[i+1] for i in range(np.shape(data.energies)[0]-1))

    data.plot_energy()

def test_energy_descreasing_async():

    '''
    This function tests if the energy function is decreasing. It may be modified in order to test all states.

    Notes
    -----
    - The function first tests the energy function by comparing values and then shows a graph of the energy states throughout updating.
    '''
    
    #Creating a system and running and saving the dynamic network
    system = SystemCreator(50, 2500, 1000)
    data = DataSaverAsync()
    network_async = HopfieldNetworkAsync(system.patterns)
    network_async.dynamics(system.perturbed_pattern, data, 10000, 1000, 0)
    
    #control if the list energy is decreasing
    assert all(data.energies[i] >= data.energies[i+1] for i in range(np.shape(data.energies)[0]-1))

    data.plot_energy()
    
def test_doctest_visual_functions(capsys):
    
    ''' Fairly silly test that runs visual_functions.py as a script '''

    import importlib.machinery
    import os
    
    path = os.path.abspath(visual_functions.__file__)
    loader = importlib.machinery.SourceFileLoader("__main__", path)
    mod = loader.load_module() 

    captured = capsys.readouterr()  #capture the output
    assert (captured.out).endswith("Finishing doctests for visual_functions.py\n") #assert the captured string ends with the expected message and not an error message for instance
