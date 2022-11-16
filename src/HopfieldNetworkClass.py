import numpy as np
from numba import jit

class HopfieldNetwork:
    '''
    The super-class HopfieldNetwork allows to initialize a Hopfield Network. That is, the instances of this class can calculate the weights needed .

    This class contains the following methods:
    - __init__
    - change_learning_rule
    - display_learning_rule
    - hebbian_weights
    - storkey_weights
    - asymptotic_capacity
    - update (abstract function that needs to be redefined)
    - dynamics (abstract function that needs to be redefined)

    And has the following attributes:
    - learning_rule
    - weights
    - capacity
    '''

    def __init__(self, patterns, learning_rule = "hebbian"):
        '''
        This method initializes the attributes of the HopfieldNetwork class.

        Parameters
        ----------
        patterns: Matrix array
            Represents a set of state of neurons networks at a given time t.
        learning_rule : string
            Represents the chosen rule to calculate the weights of the given set of patterns.
            Should be either "hebbian" or "storkey" to correspond to the Hebbian rule and Storkey rule.
        '''

        if learning_rule == "hebbian": self.weights = self.hebbian_weights(patterns)
        elif learning_rule == "storkey": self.weights = self.storkey_weights(patterns)
        else: raise SyntaxError("ERROR: There is no predefined learning rule named '{}'. Either use 'hebbian' or 'storkey'.".format(learning_rule))

        self.learning_rule = learning_rule
        network_size = patterns[0].size
        self.capacity = self.__asymptotic_capacity(network_size)
        
    def display_learning_rule(self):
        '''
        Display the attribute learning_rule.
        '''
        print("Learning rule used by the Hopkin Network: ", self.learning_rule)

    def hebbian_weights(self, patterns):
        '''
        This method creates the Hebbian weight matrix computed from attribute pattern. This matrix gives an idea of the activity certain neurons are in the network and which neurons often interact with each other.

        Parameters 
        ----------
        patterns: Matrix array
            Represents a set of state of neurons networks at a given time t.

        Returns
        -------
        Weight_matrix: matrix array
            The computed Hebbian weight matrix.

        Examples
        --------
        >>> HopfieldNetwork(np.array([[1,1,-1,-1],[1,1,-1,1], [-1,1,-1,1]]), "hebbian").weights
        array([[ 0.        ,  0.33333333, -0.33333333, -0.33333333],
               [ 0.33333333,  0.        , -1.        ,  0.33333333],
               [-0.33333333, -1.        ,  0.        , -0.33333333],
               [-0.33333333,  0.33333333, -0.33333333,  0.        ]])
        '''
        num_patterns, pattern_size = patterns.shape
        weight_matrix = np.zeros([pattern_size,pattern_size])

        for pattern in patterns:
            weight_matrix += np.outer(pattern, pattern)

        #The diagonal is made of ones (every outer product adds a diagonal of ones). This nulls the diagonal.
        np.fill_diagonal(weight_matrix,0)

        return weight_matrix/num_patterns
    
    @jit(forceobj = True)
    def storkey_weights(self, patterns):
        '''
        This method creates the Storkey weight matrix computed from the attribute patterns. The mathematical model behind this matrix allows a network to memorize patterns.
        
        Parameters 
        ----------
        patterns: Matrix array
            Represents a set of state of neurons networks at a given time t.

        Returns
        -------
        Weight_matrix: matrix array
            The Storkey weight matrix computed.

        Examples
        --------
        >>> HopfieldNetwork(np.array([[1,1,-1,-1],[1,1,-1,1], [-1,1,-1,1]]), "storkey").weights
        array([[ 1.125,  0.25 , -0.25 , -0.5  ],
               [ 0.25 ,  0.625, -1.   ,  0.25 ],
               [-0.25 , -1.   ,  0.625, -0.25 ],
               [-0.5  ,  0.25 , -0.25 ,  1.125]])
        '''

        num_patterns, pattern_size = patterns.shape
        weight_matrix = np.zeros([pattern_size,pattern_size])
            
        for pattern in patterns:

            #Computing the matrix h needed to compute the weights matrix using matrix multiplication (and substracting the diagonal off the multiplied matrix to dismiss values where j = k = i)
            pattern_matrix = np.tile(pattern, (pattern_size, 1)).T
            matrix_h = np.matmul(weight_matrix - np.diag(np.diag(weight_matrix)), pattern_matrix - np.diag(np.diag(pattern_matrix)))

            #Mathemical implementation of the Storkey learning rule
            add_to_weights = (np.outer(pattern, pattern) - (matrix_h * pattern).T - (matrix_h * pattern))
            weight_matrix += add_to_weights/pattern_size

        return weight_matrix

    def __asymptotic_capacity(self, network_size):
        '''
        This method computes the asymptotic bound of the capacity of the network according to its learning rule.

        Parameters
        ----------
        patterns : Matrix array
            Represents a set of state of neurons networks at a given time t.
        
        Returns
        -------
        capacity: interger
            Storage capacity of the neurons network.
        
        Examples
        --------
        >>> HopfieldNetwork(np.array([[1,1,-1,-1],[1,1,-1,1], [-1,1,-1,1]]), "hebbian").capacity
        1.4426950408889634
        >>> HopfieldNetwork(np.array([[1,1,-1,-1],[1,1,-1,1], [-1,1,-1,1]]), "storkey").capacity
        2.4022448175728996
        '''

        if self.learning_rule == 'hebbian':
            return network_size / (2 * np.log(network_size))
        elif self.learning_rule == 'storkey':
            return network_size / (np.sqrt(2 * np.log(network_size)))

    def update(self):
        raise NotImplementedError() #Sub-classes must define the update function
    
    def dynamics(self):
        raise NotImplementedError() #Sub-classes must define the dynamics function

class HopfieldNetworkSync(HopfieldNetwork):
    '''
    This class is a subclass of the HopfieldNetwork class, adding functionalitites to update the state of the network.
    Instances of its class will make their patterns evolve synchronously.

    This class contains the following methods:
    - update
    - dynamics
    '''

    def update(self, state):
        '''
        Updating of the state. All the elements of the state are updated by being multiplied to the weight matrix. The result is then set equal to -1 if the element is <0 and set to +1 of the element is >0.

        Parameters
        ----------
        state : 1-D array
            Pattern current state.
       
        Returns
        -------
        new state : 1-D array
            The updated state.

        Examples
        --------
        >>> HopfieldNetworkSync(np.array([[1,1,-1,-1],[1,1,-1,1], [-1,1,-1,1]]), "hebbian").update(np.array([-1, -1, 1, 1]))
        array([-1, -1,  1, -1])
        '''
        return  2*(self.weights.dot(state) > 0) - 1

    def dynamics(self, state, saver, max_iter = 20):
        '''
        Evolution of the states.
        We run the dynamical system from an initial state until convergence or until a maximum number of steps by updating all the components of the state.

        Paramters
        ---------
        state : 1-D array
            Pattern current state.
        saver : DataSaver object
            Save the state history.
        max_iter : integer
            Maximum number of steps allowed before it is considered convergence will not occur.

        '''
        #verification of the type of number of iterations
        if type(max_iter) != int :
            raise TypeError('maximum iteration must be an integer.')
        
        for i in range(max_iter) :
            
            saver.store_iter(state, self.weights)
            state_new = self.update(state)

            if np.all(state == state_new):
                saver.store_iter(state_new, self.weights)
                break
        
            state = state_new

class HopfieldNetworkAsync(HopfieldNetwork):
    '''
    This class is the subclass of the HopfieldNetwork class, adding functionalitites to update the state of the network.
    Instances of its class will make their patterns evolve asynchronously.

    This class contains the following methods:
    - update
    - dynamics
    '''
    def update(self, state):
        '''
         Updating of a state element. We update an element of the state using the rule used in function update(), but applied to a single element. We randomly choose the updated element.

         Parameters
         ----------
         state : 1-D array
            Pattern current state.
        
         Returns
         -------
         new state : 1-D array
            The state with the updated element.
        '''

        #Random choice of the changed element of the pattern
        index_update = np.random.randint(state.shape[0])
        #Update of the element to be updated (other elements remain the same). We don't want the state in parameter to change (mutable object), so we work on a copy.
        new_state = state.copy()
        new_state[index_update] = 2 * (np.dot(self.weights[index_update], new_state) > 0) - 1

        return new_state
    
    def dynamics(self, state, saver, max_iter=20000, convergence_num_iter=1000, skip = 0):
        '''
        Evolution of the states. We run the dynamical system from an initial state until convergence or until a maximum number of steps by updating all the components of the state.

        Parameters
        ----------
        state : 1-D array
            Pattern current state.
        saver : DataSaver object
            To save the state history each skip state.
        max_iter : integer
            Maximum number of steps allowed before it is considered convergence will not occur.
        convergence_num_inter : integer
             Number of steps during which state must not change to assume this is a convergence.
        skip : integer
            Instead of saving all the values, only every skip-th value will be saved
        '''

        #verification of the type of max_iter and convergence_num_iter
        if type(max_iter) != int :
            raise TypeError('maximum iteration must be an integer.')
        if type(convergence_num_iter) != int :
            raise TypeError('convergence_num_iter must be an integer.')

        skip_counter = skip
        consecutive_same_value = 0
        
        for i in range(max_iter):

            if skip_counter == skip:
                saver.store_iter(state, self.weights)
                skip_counter = 0
            else:
                skip_counter += 1
            
            state_new = self.update(state)

            if (state == state_new).all():
                consecutive_same_value += 1
            else:
                consecutive_same_value = 0

            if consecutive_same_value >= convergence_num_iter:
                saver.store_iter(state_new, self.weights)
                break
            
            state = state_new

if __name__ == "__main__":
    import doctest
    print("Starting doctests for HopfieldNetworkCLass.py")
    doctest.testmod()
    print("Finishing doctests for HopfieldNetworkClass.py")