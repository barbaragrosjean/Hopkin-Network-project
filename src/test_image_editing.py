import pytest
from visual_functions import visualize_pattern
from test_functions import pattern_check
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

def test_load_image():
    '''
    This function allows to test if an image saved in the images folder can be opened properly.
    '''
   
    system = PhotoSystemCreator("test_image.jpg", 20, "random")
    system.image.show()

    with pytest.raises(ValueError):
        PhotoSystemCreator("image_that_does_not_exist.png", 20, "random")

def test_bw_image():
    '''
    This function allows to test if an opened image can be turned into a black and white image and be downsampled respecting some size constraints.
    '''
    system = PhotoSystemCreator("test_image.jpg", 20, "random")
    system.bw_image.show()
    width, height = system.dimensions

    assert width * height <= system.max_area

def test_image_to_pattern():
    '''
    Test function to check whether the image has been well converted into a pattern fitting pattern characteristics.
    '''

    system = PhotoSystemCreator("test_image.jpg", 20, "random")
    pattern_check(system.base_pattern)
    assert system.system_pattern_size() == len(system.base_pattern)

def test_random_perturb():

    system = PhotoSystemCreator("test_image.jpg", 20, "random")
    
    visualize_pattern(system.reshape_pattern(system.base_pattern), 'base pattern')
    visualize_pattern(system.reshape_pattern(system.perturbed_pattern), 'randomly perturbed pattern')

    assert np.count_nonzero(system.base_pattern - system.perturbed_pattern) == system.system_num_perturb()

def test_corner_perturb():

    system = PhotoSystemCreator("test_image.jpg", 50, "corner")
    visualize_pattern(system.reshape_pattern(system.perturbed_pattern), 'corner perturbed pattern')

def test_invert_perturb():

    system = PhotoSystemCreator("test_image.jpg", 20, "invert")
    visualize_pattern(system.reshape_pattern(system.perturbed_pattern), 'invert perturbed pattern')

    assert np.count_nonzero(system.base_pattern + system.perturbed_pattern) == 0

def test_user_defined_perturb():

    system = PhotoSystemCreator("base_123.png", 20, "perturbed_123.png")
    visualize_pattern(system.reshape_pattern(system.base_pattern), 'base pattern')
    visualize_pattern(system.reshape_pattern(system.perturbed_pattern), 'user_defined perturbed pattern')
