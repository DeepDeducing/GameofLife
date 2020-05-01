import numpy as np
from scipy.special import expit


"""
----------------------------------------------------------------------------------------------------------------------------------------------------------
                                                       Notes for readers
Thank you for reading this note. This code is the deducing phase for deep deducing for leveraging sets of weight matrix to perform MWM-
SGD to play Conway's Game of Life.

You may change or tune any of the following parameters or variables. However, it is recommended that you do so only if the following 
note suggests so.

We hope you enjoy it.
----------------------------------------------------------------------------------------------------------------------------------------------------------
"""


"""
----------------------------------------------------------------------------------------------------------------------------------------------------------
                                                       Part A. Functions for Generating Pre-activated Inner Values of the Missing Input Data for Each Cell, etc. 
----------------------------------------------------------------------------------------------------------------------------------------------------------
"""


def generate_inner(inner, initial_state):

    missing_input_inner = list()

    back_ground         = np.zeros((initial_state.shape[0] + 2, initial_state.shape[1] + 2))

    back_ground[1 : 1 + initial_state.shape[0], 1 : 1 + initial_state.shape[1]] = initial_state[0 : initial_state.shape[0], 0 : initial_state.shape[1]]

    for i in range(initial_state.shape[0]):
        for j in range(initial_state.shape[1]):
            sequence    = (np.random.random(9) - 0.5) * 0.1 - inner * ( (back_ground[1+i-1 : 1+i+2, 1+j-1 : 1+j+2].flatten()) * 2 - 1)
            sequence[4] = (np.random.random(1) - 0.5) * 0.1 + 0
            missing_input_inner.append( sequence )
    missing_input_inner = np.array(missing_input_inner)

    return missing_input_inner


"""
----------------------------------------------------------------------------------------------------------------------------------------------------------
                                                       Part B. Starting Generations
----------------------------------------------------------------------------------------------------------------------------------------------------------
"""


initial_state = np.array\
([
# Referring to the initial state of the Game of Life table. We highly recommend readers to play with this initial state to see what will happen.
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],


])

# Referring to times of test. we recommend readers to change this parameter.
generation = 2000


for i in range(generation):


    """
    -----------------------------------------------------------------------------------------------------------------------------------------------------
                                                               Part B-1. Loading the Trained Sets of Weight Matrix for MWM-SGD
    -----------------------------------------------------------------------------------------------------------------------------------------------------
    """
    # we recommend readers to try different size of sets of weight matrix by deleting some of the weight or slope list to grasp a feeling how
    # size of sets of weight matrix affect overall accuracy and variability.
    weight_lists = list()
    slope_lists  = list()

    weight_list        = np.load("self.Conway_1_100x100x100_30_0.000001_50m_[1]_weight_list.npy" , allow_pickle=True)
    slope_list         = np.load("self.Conway_1_100x100x100_30_0.000001_50m_[1]_slope_list.npy"  , allow_pickle=True)
    weight_lists.append(weight_list)
    slope_lists.append(slope_list)

    weight_list        = np.load("self.Conway_1_100x100x100_30_0.000001_50m_[2]_weight_list.npy" , allow_pickle=True)
    slope_list         = np.load("self.Conway_1_100x100x100_30_0.000001_50m_[2]_slope_list.npy"  , allow_pickle=True)
    weight_lists.append(weight_list)
    slope_lists.append(slope_list)

    weight_list        = np.load("self.Conway_1_100x100x100_30_0.000001_50m_[3]_weight_list.npy" , allow_pickle=True)
    slope_list         = np.load("self.Conway_1_100x100x100_30_0.000001_50m_[3]_slope_list.npy"  , allow_pickle=True)
    weight_lists.append(weight_list)
    slope_lists.append(slope_list)

    weight_list        = np.load("self.Conway_1_100x100x100_30_0.000001_50m_[4]_weight_list.npy" , allow_pickle=True)
    slope_list         = np.load("self.Conway_1_100x100x100_30_0.000001_50m_[4]_slope_list.npy"  , allow_pickle=True)
    weight_lists.append(weight_list)
    slope_lists.append(slope_list)


    """
    -----------------------------------------------------------------------------------------------------------------------------------------------------
                                                               Part B-2. Initializing Pre-activated Inner Values of the Missing Input Data for Each Cell, etc.
    -----------------------------------------------------------------------------------------------------------------------------------------------------
    """


    inner                  = -3.50

    missing_input_inner    = generate_inner(inner, initial_state)

    missing_input_resistor = np.ones_like(missing_input_inner) * np.array([0, 0, 0, 0, 1, 0, 0, 0, 0])

    desired_output         = np.ones((missing_input_inner.shape[0], missing_input_inner.shape[1]))


    """
    -----------------------------------------------------------------------------------------------------------------------------------------------------
                                                               Part B-3. Importing Model
    -----------------------------------------------------------------------------------------------------------------------------------------------------
    """


    from Brain_for_deducing import *

    network_size                = np.array([9, 100, 100, 100, 9])
    # Referring to deducing rate. we recommend readers to change this parameter.
    beta                        = 0.1
    # Referring to deducing epochs "between each mandatory pulse". we recommend readers to change this parameter.
    epoch_of_deducing           = 10000
    # Referring to the rate of neurons dropped out in the hidden layers. For example, if drop_rate = 0.2, it means 20% of the neurons in the hidden layers will be dropped out on a random base.  we recommend readers to change this parameter.
    drop_rate                   = 0.1

    Machine                     = Brain(network_size, beta, epoch_of_deducing, drop_rate)


    """
    -----------------------------------------------------------------------------------------------------------------------------------------------------
                                                               Part B-4-1. Back-propagation and WMW-SGD
    -----------------------------------------------------------------------------------------------------------------------------------------------------
    """


    for j in range(Machine.epoch_of_deducing ):


        random_index     = np.random.randint(np.array(weight_lists).shape[0])
        weight_list      = weight_lists[random_index]
        slope_list       = slope_lists[random_index]


        missing_input_inner                           = Machine.deduce_batch(missing_input_inner,
                                                                             missing_input_resistor,
                                                                             desired_output,
                                                                             weight_list,
                                                                             slope_list
                                                                             )


    """
    -----------------------------------------------------------------------------------------------------------------------------------------------------
                                                               Part B-4-2. Mandatory Pulse
    -----------------------------------------------------------------------------------------------------------------------------------------------------
    """


    sequence  = missing_input_inner[:, 4].flatten()

    new_state = np.zeros_like(sequence)
    for j in range(sequence.shape[0]):
        if   sequence[j] >= 1.0:
            new_state[j] = 1
        elif sequence[j] <= -1.0:
            new_state[j] = 0
        else:
            new_state[j] = initial_state.flatten()[j]
    new_state = new_state.reshape((initial_state.shape[0], initial_state.shape[1]))
    new_state = np.array(new_state, dtype=int)


    """
    -----------------------------------------------------------------------------------------------------------------------------------------------------
                                                               Part B-5. Printing out the Next Generation of the Table Proposed by the Machine, etc.
    -----------------------------------------------------------------------------------------------------------------------------------------------------
    """


    print("Generation -------------------------------------------------------------")
    print(i)
    print("------------------------------------------------------------------------")
    print(new_state)


    if (1 in new_state[:, -1]) | (1 in new_state[-1, :]):
        new_state = np.vstack((  new_state, np.zeros((5, new_state.shape[1]))  ))
        new_state = np.hstack((  new_state, np.zeros((new_state.shape[0], 5))  ))
        new_state = new_state[4:-1, 4:-1]
        initial_state     = new_state
    else:
        initial_state     = new_state

    #initial_state     = new_state

