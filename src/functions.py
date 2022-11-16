import numpy as np

'''Overview of functions contained in this module:
     - pattern_match
     - verify_convergence
'''
    
def pattern_match(memorized_patterns, pattern) :

    '''
    Controls if a pattern matches with one of the memorized patterns. In a list of memorized patterns we search if the pattern in parameters matches with any of them. 

    Parameters
    -----------
    memorized_patterns : list of arrays
        Contain all the pattern memorized.
    pattern :  1-D array
        The pattern which is being tested for being equal to one of the memorized patterns.

    Returns
    ------
    i : integer
        index of the first matching pattern in memorized_patterns
    None if there no pattern matches.

    Example
    -------
    >>> pattern_match(np.array([[ 1.,  1., -1., -1.], [ 1.,  1., -1., -1.], [-1., -1.,  1.,  1.]]), [ 1.,  1., -1., -1.])
    0
    >>> pattern_match(np.array([[ 1.,  1., -1., -1.], [ 1.,  1., -1., -1.], [-1., -1.,  1.,  1.]]), [ 1.,  1.,  1.,  1.])
    '''
    
    for i in range(memorized_patterns.shape[0]) :
        if (pattern == memorized_patterns[i]).all() :
            return i
          
def verify_convergence(base_pattern, last_state, print_found = False):

    '''
    Function used to verify whether the state found by running the system is the same as the starting pattern.

    Parameters
    ----------
    base_pattern: array
        Pattern before perturbation and update (the pattern our system should be able to find if functional)
    last_state: array
        Last pattern found by updating the base pattern

    Returns
    -------
    found: bool
        Truth value indicating whether the initial pattern has been found after updating of the perturbed pattern
    '''

    if np.all(last_state == base_pattern):
        if print_found: print("\nUpdating the perturbed pattern has allowed to retrieve base pattern.")
        return True
    else: 
        if print_found: print("\nUpdating the perturbed pattern has not allowed to retrieve base pattern.")
        return False
    
if __name__ == "__main__":
    import doctest
    print("Starting doctests for functions.py")
    doctest.testmod()
    print("Finishing doctests for functions.py")
