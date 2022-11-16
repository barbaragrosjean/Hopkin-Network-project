import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.animation as animation
import numpy as np
import os

'''
This file contains three classes : DataSaver, DataSaverSync, DataSaverAsync. DataSaverSync and DataSeverAsync inherit from DataSaver. That permit to have specificity for the energy plotting and the reshape function.
'''

class DataSaver:
    '''
    This class contains the following methods:
    - __init__
    - reset
    - store_iter
    - compute_energy
    - get_data
    - create_video
    - save_video
    - show_video
    - reshape_patterns
    - plot_energy

    And has the following attributes:
    - states
    - energies
    - dimensions
    '''

    def __init__(self, dimensions = (0,0)):
        '''
        Initialisation of the DataSaver instance : empty list of states, empty list of energy, dimensions of working with dimensions is relevant.
        '''
        self.reset()
        self.dimensions = dimensions
             
    def reset(self):
        '''
        This functions allows to reset a DataSever. it gives empty lists for states and energies and reset.
        '''
        
        self.states = []
        self.energies = []
             
    def store_iter(self, state, weights):
        '''
        This function allows to store for a given state and weight the state itself and its energy in our lists of data.
        Parameters
        ----------
        state : 1-D array
            Pattern of current state.
        weights : 2-D array
            Weights of the system. Represent the connection state between each neuron.
        '''
        self.states.append(state)
        self.energies.append(self.compute_energy(state, weights))
        
    def compute_energy(self, state, weights):
        '''
        Computation of the energy of a given state.

        Parameters
        ----------
        state : 1-D array
            Pattern of current state.
        weights : 2-D array
            Weights of the system. Represent the connection state between each neuron.
        '''
        
        return -np.dot(np.matmul(weights,state), state)/2
        
    def get_data(self):
        '''
        This function return the attributes of the current instance.
        
        Returns
        -------
        state : list of states
        energy : list of energies
        '''
        return self.state, self.energy
        
    def create_video(self, img_shape = 5, title = 'System Evolution'):
        '''
        Creation of an animation in which we see the system evolution.

        Parameters
        ----------
        title = 'System Evolution' : string
            Title the animation will have
        
        Returns
        -------
        animation : ArtistAnimation
            video containing the consecutive state computed by the system.
        img_shape : integer
            image dimension

        Note
        ----
        This function creates a plot when called. To avoid having many unseen plots (that all open the next time plt.show() is called), use this function only together with save_video or show_video.
        '''
        reshaped_states = self.reshape_states()
        
        if len(reshaped_states[0].shape) == 1:
            raise ValueError("The states must be reshaped as square matrices")

        list_im = []
        cmap = colors.ListedColormap(['k','w'])
        fig = plt.figure(figsize=(img_shape, img_shape))
        plt.title(title)

        #list of states become a list of frames : imshow()
        for state in reshaped_states :
            list_im.append([plt.imshow(state,cmap)])
        
        #create an animation
        return animation.ArtistAnimation(fig, list_im)
    
    def save_video(self, out_path):
        '''
        Saving of a video where we see the system evolution. The video is saved with the file name given in out_path and a .gif extension, in the folder /animations starting from the base folder of the project.

        Parameters
        ----------
        out_path :
            name with which the video is saved, starting from root folder of the project and changing in directory /animations.
        img_shape : tuple
            
        '''
        
        path = os.path.join(os.path.dirname(__file__), os.pardir)  #parent directory of src
        path = os.path.join(path, 'animations')                    #going into new directory animations
        os.makedirs(path, exist_ok=True)                           #creating new directory animations if needed
        filename = out_path + '.gif'

        writergif = animation.PillowWriter(fps = 3)
        self.create_video().save(os.path.join(path, filename), writer = writergif)
        
        print("File {} saved under: ".format(filename), path, "\n")
        plt.close() #Closing the plot so it doesn't open next time plt.show() is called
        
    def show_video(self) :

        '''
        Showing of a video where we see the system evolution in a pop-up window.

        Parameters
        ----------
        title : string
            Title the displayed animation will have
        '''
        animations = self.create_video()
        plt.show()
    
    def reshape_states(self) :
        raise NotImplementedError
     
    def plot_energy(self):
        '''
        Function used to visualize the properties of the energy function, which should always be non-increasing. Prints a graph displaying the energy for a number of states along the update process.
        '''
        
        #Defining the x values for the graph
        x = np.arange(len(self.energies))
        
        #Computing values, assigning labels, title and subplot characteristics
        plt.plot(x, self.energies, color ='navy', marker = '.')
        if isinstance(self, DataSaverSync): plt.title('Synchronous update')
        elif isinstance(self, DataSaverAsync): plt.title('Asynchronous update')
        plt.xlabel('Number of the state in update history')
        plt.ylabel('Energy')
        
        plt.show()

class DataSaverSync(DataSaver) :
    
    def reshape_states(self):
        '''
        Transforms linear states into matrices that have the dimensions given by the attribute dimensions.
            
        Returns
        -------
        states_reshaped : matrix
        '''

        height, width = self.dimensions

        #Sample test: if the first array has the right size, we consider that the following ones will have the right size too
        if len(self.states[0]) != height * width :
            raise ValueError("The dimensions to reshape the states are incompatible with their length")

        states_reshape = []
    
        for state in self.states :
            states_reshape.append(np.reshape(state, (width, height)))
        
        return states_reshape
            
class DataSaverAsync(DataSaver) :
    
    def reshape_states(self):

        '''
        Transforms linear states into square matrices (if the dimension allows it, i.e. is the lengths of a pattern is a squared integer). For instance, a pattern of length 25 will be reshaped as a 5x5 matrix.

        Returns
        -------
        states reshaped : n x n matrix

        '''
        height, width = self.dimensions

        #Sample test: if the first array has the right size, we consider that the following ones will have the right size too
        if len(self.states[0]) != height * width :
            raise ValueError("The dimensions to reshape the states are incompatible with their length")

        states_reshape = []

        for i in np.linspace(0, np.sqrt(len(self.states)-1), 15, dtype=int):
            states_reshape.append(np.reshape(self.states[i**2], (width, height)))

        return states_reshape
