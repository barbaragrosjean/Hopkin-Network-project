import numpy as np
from math import floor
from PIL import Image, ImageOps
import os

class SystemCreator:
    '''
    This class allows to create a system used by a network. Such a system consists of a set of patterns, of which one pattern is chosen as base pattern and perturbed.

    This class contains the following methods:
    - __init__
    - generate_patterns
    - perturb_patterns 
    - generate_system
    - generate_visual_system

    And has the following public attributes:
    - patterns
    - base_pattern
    - perturbed_pattern
    - dimensions
    '''

    def __init__(self, num_patterns, size_patterns, num_perturb, use_for_visualization = False, checkerboard_submatrix_size = 5):
        '''
        This method initializes the attributes of an instance of the SystemCreator class.

        Parameters
        ----------
        num_patterns: integer
            Number of patterns memorized by the network
        size_patterns: integer
            Size of all patterns memorized by the network
        num_perturb: integer
            Number of elements to be perturbed in the initial pattern
        use_for_visualization: bool
            Allows to indicate whether the patterns that will be create are meant to be visualizable or not
        checkerboard_submatrix_size: int
            Allows to define the size we want the checkerboard's (pattern for visualization) black and white submatrices to have.
        '''

        if use_for_visualization: 
            self.generate_visual_system(num_patterns, size_patterns, num_perturb, checkerboard_submatrix_size)
            self.use_for_visualization = True
        else: 
            self.generate_system(num_patterns, size_patterns, num_perturb)
            self.use_for_visualization = False

        self.__num_perturb = num_perturb

    @staticmethod
    def generate_patterns(num_patterns, pattern_size):
        '''
        This function generates patterns representing the state of a neuron network at a moment t. A pattern is represented by an array of dimension N, where N is the number of neurons in the network. Every element of the array is equal to either 1 or -1, the first indicating that the neuron fires and the second that it does not fire.

        Parameters
        ----------
        num_patterns: integer
            The number of patterns
        pattern_size: integer
            The size of the patterns

        Returns
        ----------
        P: matrix array
            It represents the set of patterns created of dimension num_patterns x pattern_size.
        '''
        #exceptions
        if type(num_patterns) != int :
            raise TypeError('the number of patterns must be an integer')
        if type(pattern_size) != int :
            raise TypeError('the pattern size must be an integer')
        
        return np.random.choice([-1,1], [num_patterns,pattern_size])

    def perturb_pattern(self, pattern, num_perturb):
        '''
        Perturbation of a pattern. This function allows to perturb num_perturb distinct elements chosen randomly in the pattern. The perturbation is defined as the switching of the value of the chosen element to the other possible value (1 becomes -1 and -1 becomes 1).

        Parameters
        ----------
        pattern : 1-D array
            Contains -1 or 1, describes the state of a neurone network.
        num_perturb : integer
            Number of elements to be switched.

        Returns
        ------
        pattern : 1-D array
            The perturbed pattern.
        '''
    
        if type(num_perturb) != int :
            raise TypeError('the number of perturbation must be an integer')
        if num_perturb > pattern.size:
            raise ValueError('the number of perturbed elements cannot exceed the number of elements of the pattern')
        
        perturbed_pattern = pattern.copy()
        indices = np.random.choice(perturbed_pattern.shape[0], num_perturb, replace=False)
        perturbed_pattern[indices] *= -1
        
        return perturbed_pattern

    def next_base_pattern(self):
        '''
        This function allows change the base pattern to the next one in the list of patterns and to perturb it. Once reached the end of the list, it start over again.
        '''
        if self.__base_pattern_index < len(self.patterns) - 1: self.__base_pattern_index += 1
        else: self.__base_pattern_index = 0

        self.base_pattern = self.patterns[self.__base_pattern_index]
        self.perturbed_pattern = self.perturb_pattern(self.base_pattern, self.__num_perturb)

    def change_num_perturb(self, new_num_perturb):
        '''
        This function allows to change the number of perturbed elements of the perturbed base pattern
        '''
        self.perturbed_pattern = self.perturb_pattern(self.base_pattern, new_num_perturb)

    def generate_system(self, num_patterns, size_patterns, num_perturb):
        '''
        This function allows to generate a group of patterns with which we can compute the weights and to isolate and perturb a pattern of interest.
        '''

        self.__base_pattern_index = 0
        self.patterns = self.generate_patterns(num_patterns,size_patterns)
        self.base_pattern = self.patterns[self.__base_pattern_index]
        self.perturbed_pattern = self.perturb_pattern(self.base_pattern, num_perturb)
        self.dimensions = (size_patterns, 0)

    def generate_visual_system(self, num_patterns, size_patterns, num_perturb, checkerboard_submatrix_size):
        '''
        This function allows to generate a group of patterns with which we can compute the weights and to isolate and perturb a pattern of interest.
        
        Note
        ----
        This functions returns patterns that can be used for visualization because their size is a perfect square (representing patterns as squares) and because the base pattern is recognizable (it is a checkerboard).
        '''

        from visual_functions import generate_checkerboard

        #Determining the largest perfect square that is tinier than the given pattern size, so patterns can be represented as a square matrix
        size_matrix = floor(np.sqrt(size_patterns))
        size_patterns = size_matrix**2
        self.dimensions = (size_matrix, size_matrix)

        #Creating patterns and substituting the first pattern with the flattened chosen checkerboard
        self.patterns = self.generate_patterns(num_patterns, size_patterns)
        self.base_pattern = self.patterns[0] = generate_checkerboard(size_matrix, checkerboard_submatrix_size).flatten()
        self.perturbed_pattern = self.perturb_pattern(self.base_pattern, num_perturb)

class PhotoSystemCreator(SystemCreator):

    '''This subclass of SystemCreator allows to create a system starting with an image. The size of the patterns result of the size of the image and multiple features to work on the image are provided.

    This subclass adds the following methods:
    - __init__
    - perturb_corner
    - perturb_invert
    - set_perturbed_pattern
    - load_image
    - make_image_bw
    - image_to_pattern
    - system_pattern_size
    - system_num_perturb
    - reshape_pattern

    It also adds attributes:
    - image
    - bw_image
    - max_area
    '''

    def __init__(self, img_path, per_perturb, perturbation):

        '''
        This initialiser allows to create patterns and perturb it in different ways (either random perturbation, keeping only a corner, inverting all pixels or choosing a manual pattern loaded from another image)
        '''

        #loading the image, downsampling to an adequate size and transforming into black and white image
        self.image = self.load_image(img_path)
        self.image = self.__downsample(self.image)
        self.bw_image = self.make_image_bw(self.image)
        self.dimensions = self.bw_image.size
        self.__per_perturb = per_perturb

        #making a set of 3 patterns with the size corresponding to the loaded image
        self.patterns = self.generate_patterns(3, self.system_pattern_size())
        self.base_pattern = self.patterns[0] = self.image_to_pattern(self.bw_image)

        if perturbation == "random": self.perturbed_pattern = self.perturb_pattern(self.base_pattern, self.system_num_perturb())
        elif perturbation == "corner": self.perturbed_pattern = self.perturb_corner()
        elif perturbation == "invert": self.perturbed_pattern = self.perturb_invert()
        else : self.perturbed_pattern = self.set_perturbed_pattern(perturbation)

    def perturb_corner(self):
        '''
        This function is made to perturb an image by blanking all the image except for a rectangle in the upper left corner.
        '''

        end_perturb = (100-self.__per_perturb)/100
        width, height = self.dimensions
        end_width = int(end_perturb * width)
        end_height = int(end_perturb * height)

        #indices inverted because numpy reshape understands dimensions differently
        perturbed_pattern = np.ones((height, width)) 
        base_pattern = self.reshape_pattern(self.base_pattern)
        perturbed_pattern[0 : end_height, 0 : end_width] = base_pattern[0 : end_height, 0 : end_width]

        return perturbed_pattern.flatten()
        
    def set_perturbed_pattern(self, perturbed_img_path):
        '''
        This function manually sets the perturbed pattern (by passing its image's path in parameters).
        '''

        #loading the image, downsampling to an adequate size and transforming into black and white image
        perturbed_image = self.load_image(perturbed_img_path)
        self.__downsample(perturbed_image)
        perturbed_image = self.make_image_bw(perturbed_image)

        if self.dimensions != perturbed_image.size: raise ValueError("The perturbed image does not have the same size as the base image")

        return self.image_to_pattern(perturbed_image)

    def perturb_invert(self):
        '''
        This function perturbs an image by inverting all the pixels (black becomes white and white becomes black).
        '''

        return -1 * self.base_pattern

    def __downsample(self, image):
        '''
        This function downsamples the picture of the sytem to a size that is appropriate for a Hopfield network (maximal dimensions are (400, 400)).
        '''

        width, height = image.size
        self.max_area = 14400

        if width * height > self.max_area:
            ratio = width / height
            dArea = self.max_area
            dimensions = (int(np.sqrt(dArea * ratio)), int(np.sqrt(dArea / ratio)))
            return image.resize(dimensions)
        
        return image

    def load_image(self, img_path):
        '''
        This function allows to load an image saved in the 'images' folder.
        '''

        filepath = os.path.join(os.path.dirname(__file__), os.pardir)  #parent directory of src
        filepath = os.path.join(filepath, 'images', img_path)          #setting path to the absolute path of the img_path in parameters

        if os.path.exists(filepath):
            print("Imaged used by the network: images\\{} \n".format(img_path))
            im = Image.open(filepath)
            im = ImageOps.exif_transpose(im)
            return im
        else:
            raise ValueError("The entered file name does not exist.")    

    def make_image_bw(self, image):
        '''
        This function allows to make the image of the class instance black and white.
        '''
        return image.convert('1')

    def image_to_pattern(self, image):
        '''
        This method can be used to transform a black and white picture into a linear vector of -1 (black) and 1 (black).
        '''
    
        return 2 * np.array(image.getdata()) // 255 - 1
        
    def system_pattern_size(self):
        '''
        This function returns the pattern size that is implicitly set by the image choice.
        '''

        width, height = self.dimensions
        return width * height

    def system_num_perturb(self):
        '''
        Returns the number of perturbations corresponding to the percentage of perturbation given a the system's pattern size.
        '''

        if self.__per_perturb < 0 or self.__per_perturb > 100:
            raise ValueError("The perturbation percentage must be between 0 and 100")

        return int(self.__per_perturb / 100 * self.system_pattern_size())

    def reshape_pattern(self, pattern):
        '''
        Numpy arrays and pillow dimensions are inverted. For our use we need to reshape by switching height and width to obtained the wanted result.
        '''

        height, width = self.dimensions
        return pattern.reshape((width, height))

