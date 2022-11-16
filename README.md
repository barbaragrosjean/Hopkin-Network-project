# Hopkin Network project
## BIO-210-team-9  

This project's goal is to create and illustrate the simulation of a neural network. This networks is able to store a certain number of 'memories' (which are represented as vectors of -1 and 1 which we call ```patterns```). Given similar states of the neural network, the system is then able to find which memorized pattern is most similar, and offers visual representation of how this state is retrieved. It also features many tests, which can be used following the procedure described below. Besides directly using a network's functionalities, the project offers possibilities to analyse the performance of such a network, by allowing to launch experiences and analyse their results. A range of experiments and their result and analysis have already been performed and are presented in the [summary file](summary.md). This version also allows to load any image of any size in the project to use for visualization. Images added in folder **images**, which is in the root folder of the project can be transformed into a pattern that can be used by our networks. This allows to then see how any perturbed image is retrieved by the neural network! 

The project mainly provides 5 classes that can be used to simulate a neuronal network, as well as a few separate functions. A brief description of the 3 classes:

- **SystemCreator**: allows to create a set of patterns, to choose one as base pattern and to perturb it. These three elements are saved as instance attributes.
- **HopfieldNetwork**: main class of the project, which contains all the methods to create and run a network.
- **DataSaver**: an instance of class HopfieldNetwork uses an instance of this class to save the states computed during updating. Besides saving the states of the network, this class also allows to visualize them and to show their corresponding energy.
- **ExperimentClass**: allows to test either capacity or robustness of a network by running several trials with different input values. The experiment outputs are then saved in the res folder as .h5 files
- **AnalyserClass**: this class mainly allows to analyse the data created by the class Experiment and stored in the .h5 data files.

# Requirements

- Python >= 3.5
- numpy
- pytest
- matplotlib
- benchmark
- numba
- Pillow
- pandas
- tables

# Example of use

To see and test the functionality of the Network, simply run file main.py on any IDE or in a terminal with the command:

```python3 main.py```

main.py contains a few functions where some usages of the different classes are shown, these examples can either directly be used or serve as guide and inspiration for other uses of these classes!

# Testing

We use [doctests](https://docs.python.org/3/library/doctest.html) and [pytest](https://docs.pytest.org/en/6.2.x/contents.html). 

## Doctests

You can (doc)test the Hopkin Network (more specifically, doctest the functions or the visual functions) by entering one of the following lines in a terminal, being in the right folder (/src from the project root folder):

```python3 functions.py
   python3 visual_functions.py
``` 

If there is only two statements printed indicating that the tests started and stopped, no prints in between and no follow-up prints, the doctests passed. 

You can also get a detailed output pass verbose flag:

```python3 functions.py -v
   python3 visual_functions.py -v
``` 

## Pytests 

We also created unit-tests for the Hopkin Network in `test_functions.py` and `test_visual_functions.py`. You can also run the whole test-suite with:

```pytest```

## Coverage 

You can assess the coverage by running:

```
coverage run -m pytest
coverage report
```

For a nicer html format, use:
```
coverage html
```

