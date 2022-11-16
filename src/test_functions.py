import numpy as np
from DataSaverClass import *
from HopfieldNetworkClass import *
from SystemCreatorClass import *
import pytest
import functions

'''Overview of functions contained in this module:
     - test_hebbian_weights
     - test_storkey_weights
     - test_generate_pattern
     - test_perturb_pattern
     - test_pattern_match
     - test_update
     - test_update_async
     - test_dynamics
     - test_convergence_sync
     - test_convergence_async
     - test_doctest_functions
     - pattern_check
     - matrix_check
'''

def test_hebbian_weights(benchmark):

    '''
    This function is made to verify whether that the hebbian weights matrix is symmetric, has zeros on the diagonal and that the values are in the range [-1,1].
    '''
    patterns = SystemCreator.generate_patterns(50, 2500)
    network = HopfieldNetwork(patterns, "hebbian")
    benchmark.pedantic(network.hebbian_weights, args = (patterns,),iterations = 5)
    matrix_check(network.weights)
    
def test_storkey_weights():

    '''
    This function is made to verify whether that the Storkey weights matrix is symmetric and if that its values (except for the diagonal) are in the range [-1,1].
    
    Note
    ----
    matrix check:
        For Storkey weights we only want to test values that are not on the diagonal therefore we set the diagonal at zero before launching the test.
    '''

    network = HopfieldNetwork(SystemCreator.generate_patterns(50, 2500), "storkey")

    #we consider a matrix with zeros on the diagonal for testing
    matrix_check(network.weights - np.diag(np.diag(network.weights)))

def test_learning_rule(capsys):
    '''
    This function tests the methods change_learning_rule and display_learning of the class HopfieldNetwork.
    '''
    
    hebbian_network = HopfieldNetwork(SystemCreator.generate_patterns(1, 3), "hebbian")
    storkey_network = HopfieldNetwork(SystemCreator.generate_patterns(1, 3), "storkey")

    # Here we test the display_learning_rule method

    def capture_display(network):                  
        network.display_learning_rule()
        captured = capsys.readouterr()                  
        return captured.out

    assert capture_display(hebbian_network).endswith("Learning rule used by the Hopkin Network: hebbian\n")
    assert capture_display(storkey_network).endswith("Learning rule used by the Hopkin Network: storkey\n")

def test_generate_patterns():

    '''Test function allowed to verify that generate_patterns does indeed generate the right amount of patterns of the right size and content.'''

    num_patterns, size_patterns = 40, 50
    patterns = SystemCreator.generate_patterns(num_patterns, size_patterns)

    for pattern in patterns:
        pattern_check(pattern)

    assert patterns.shape == (num_patterns, size_patterns)

    #Testing if raising exceptions works
    with pytest.raises(TypeError):
        SystemCreator.generate_patterns('a', 1)
    with pytest.raises(TypeError):
        SystemCreator.generate_patterns(0,1.0)

def test_perturb_pattern():

    '''
    This function allows to test whether a pattern has been well perturbed.
    '''

    num_perturb = 3
    system = SystemCreator(1, 10, num_perturb)
    pattern_check(system.perturbed_pattern) #Check if the returned pattern fits the characteristics of a pattern
    
    #Counts how many of the patterns elements have changed
    assert np.count_nonzero(system.base_pattern - system.perturbed_pattern) == num_perturb

    #Testing if raising exceptions works
    with pytest.raises(TypeError):
        system.perturb_pattern(system.base_pattern, system.base_pattern)
    with pytest.raises(ValueError):
        system.perturb_pattern(system.base_pattern, num_perturb + 100)

def test_pattern_match():

    '''
    This function allows to test whether pattern_match is able to find an array in a collection of arrays.
    '''

    patterns = SystemCreator.generate_patterns(50,50)
    indices = np.random.choice(50, 25, replace=False)
    
    for index in indices:
        assert functions.pattern_match(patterns, patterns[index]) == index

def test_update(benchmark):

    '''This test function allows to verify whether the pattern returned by updating fits the characteristics of a pattern'''
    
    system = SystemCreator(50, 2500, 1000)
    network = HopfieldNetworkSync(system.patterns)

    updated_pattern = benchmark.pedantic(network.update, args = (system.perturbed_pattern,), iterations = 100)
    
    pattern_check(updated_pattern)
    
def test_update_async(benchmark):

    '''This test function allows to verify whether at most one component has changed during a random update of one element of an array'''
   
    system = SystemCreator(50, 2500, 1000)
    network = HopfieldNetworkAsync(system.patterns)
   
    updated_pattern = benchmark.pedantic(network.update, args = (system.perturbed_pattern,), iterations = 5)
    
    pattern_check(updated_pattern)
    assert np.count_nonzero(system.perturbed_pattern - updated_pattern) <= 1

def test_dynamics():

    '''Test function allowing to test whether the result of a sequence of synchronous updates follows some conditions:
       - if the system converged (i.e. it took less than the maximum number of steps to finish), there must be a certain number of identical states in the states list.
       - if the system did not converge, the length of the states list must be exactly the maximum number of steps to finish
       
       There are thus different scenarios tested:
       1. The setup makes it easy for the system to converge
       2. Impossible for the system to converge
       3. Wrong entries to test exceptions
    '''
    
    system = SystemCreator(25, 2500, 100, True)
    network = HopfieldNetworkSync(system.patterns)

    def easy_test():

        '''1. The setup makes it easy for the system to converge'''

        max_iter_sync = 20
        data = DataSaver()
        network.dynamics(system.perturbed_pattern, data, max_iter_sync)

        #Testing if the state history is not longer than allowed. Withdrawing the initial state before updating started
        len_states = len(data.states) - 1
        assert max_iter_sync >= len_states

        #If the system converged, test that at least 2 elements.
        if max_iter_sync > len_states: assert np.all(data.states[-2] == data.states[-1])

    def hard_test():
        
        '''2. Impossible for the system to converge'''

        max_iter_sync = 0
        data = DataSaver()
        network.dynamics(system.perturbed_pattern, data, max_iter_sync)

        #Testing if the state history is not longer than allowed. Withdrawing the initial state before updating started
        len_states = len(data.states) - 1
        assert max_iter_sync >= len_states

        #If the system did not converge, test that the last states are indeed not equal
        if max_iter_sync == len_states: assert np.all(data.states[-2] != data.states[-1])
       

    def error_test():

        '''3. Wrong entries to test exceptions'''

        data = DataSaver()

        #Testing if raising exceptions works
        with pytest.raises(TypeError):
            network.dynamics(system.perturbed_pattern, data, network.weights, 'a')
    
    easy_test(), hard_test(), error_test()

def test_dynamics_aync():

    '''Test function allowing to test whether the result of a sequence of asynchronous updates follows some conditions:
       - if the system converged (i.e. it took less than the maximum number of steps to finish), there must be a certain number of identical states in the states list.
       - if the system did not converge, the length of the states list must be exactly the maximum number of steps to finish
       
       There are thus different scenarios tested:
       1. The setup makes it easy for the system to converge
       2. Impossible for the system to converge
       3. Wrong entries to test exceptions
    '''

    system = SystemCreator(25, 2500, 100, True)
    network = HopfieldNetworkAsync(system.patterns)

    def easy_test():

        '''1. The setup makes it easy for the system to converge'''

        max_iter_async, convergence_num_iter = 30000, 1000
        data = DataSaver()
        network.dynamics(system.perturbed_pattern, data, max_iter_async, convergence_num_iter, 0)

        #Testing if the state history is not longer than allowed. Withdrawing the initial state before updating started
        len_states = len(data.states) - 1
        assert max_iter_async >= len_states

        #If the system converged, convergence_num_iter elements are the same.
        if max_iter_async > len_states: 
            assert (np.all(data.states[index], data.states[-1]) for index in range(-convergence_num_iter, 0))

    def hard_test():
        
        '''2. Impossible for the system to converge'''

        max_iter_async, convergence_num_iter = 30, 10
        data = DataSaver()
        network.dynamics(system.perturbed_pattern, data, max_iter_async, convergence_num_iter, 0)

        #Testing if the state history is not longer than allowed. Withdrawing the initial state before updating started
        len_states_async =  len(data.states) - 1
        assert max_iter_async >= len_states_async

        #If the system did not converge, test that the last states are indeed not equal.
        if max_iter_async == len_states_async: 
            assert (not np.all(data.states[index], data.states[-1]) for index in range(-convergence_num_iter, 0))
       

    def error_test():

        '''3. Wrong entries to test exceptions'''

        data = DataSaver()

        #Testing if raising exceptions works.
        with pytest.raises(TypeError):
            network.dynamics(system.perturbed_pattern, data , 'a', 100, 0)
        with pytest.raises(TypeError):
            network.dynamics(system.perturbed_pattern,data, 100, 'a', 0)
    
    easy_test(), hard_test(), error_test()

def test_convergence(capsys):

    '''
    This function tests if we obtain at the end of the simulation the same pattern as the base pattern.
    
    Note 
    ----
    The function may be changed for the test to compute the weight with Hebbian or Storkey rules.
    '''

    system = SystemCreator(100, 2500, 100)
    network = HopfieldNetworkSync(system.patterns)
    saver = DataSaverSync()
    network.dynamics(system.perturbed_pattern, saver, 20)
    
    found_pattern = functions.verify_convergence(system.base_pattern, saver.states[-1], True)
    assert found_pattern

    captured = capsys.readouterr()  # capture the output
    #check whether the captured output ends with the string confirming that the base pattern has been found
    searched_string = "\nUpdating the perturbed pattern has allowed to retrieve base pattern.\n"
    assert (captured.out).endswith(searched_string)
    
def test_doctest(capsys):
    
    ''' Slightly silly test that runs functions.py as a script '''
    
    import HopfieldNetworkClass
    import importlib.machinery
    import os

    path = os.path.abspath(functions.__file__)
    loader = importlib.machinery.SourceFileLoader("__main__", path)
    mod = loader.load_module() 

    captured = capsys.readouterr()  # capture the output
    assert (captured.out).endswith("Starting doctests for functions.py\nFinishing doctests for functions.py\n") #assert the captured string ends with the expected message and not an error message for instance

    path = os.path.abspath(HopfieldNetworkClass.__file__)
    loader = importlib.machinery.SourceFileLoader("__main__", path)
    mod = loader.load_module() 

    captured = capsys.readouterr()  # capture the output
    assert (captured.out).endswith("Starting doctests for HopfieldNetworkCLass.py\nFinishing doctests for HopfieldNetworkClass.py\n") #assert the captured string ends with the expected message and not an error message for instance

def pattern_check(pattern) :
    
    '''
    This function allows to verify the features of any pattern that needs to be checked. These features are: pattern contains values that are either -1 and 1, the shape is a one-dimensional array of type numpy array.

    Parameters
    ----------
    pattern : array
        the tested element
    '''
    assert np.any(np.abs(pattern) == 1)
    assert type(pattern) == np.ndarray
    assert len(np.shape(pattern)) == 1

def matrix_check(matrix):

    '''
    This function is made to verify whether a matrix is symmetric, has zeros on the diagonal and that the values are in the range [-1,1].
    
    Parameters
    ----------
    matrix: array of arrays
        The matrix that needs to be tested for the above listed criteria
    '''
    
    #Verify that the diagonal is composed of 0
    assert np.allclose(np.diagonal(matrix), 0)
    #Verify that the matrix is symetric
    assert np.allclose(matrix, matrix.T)
    #Verify that the matrix is in the range [-1,1]
    assert np.all(np.absolute(matrix)<=1)
