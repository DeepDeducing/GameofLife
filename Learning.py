import numpy as np


"""
----------------------------------------------------------------------------------------------------------------------------------------------------------
                                                       Notes for readers
Thank you for reading this note. This code is the learning phase for deep deducing for generating sets of weight matrix to be randomly 
selected in the deducing phase.

You may change or tune any of the following parameters or variables. However, it is recommended that you do so only if the following 
note suggests so.

We hope you enjoy it.
----------------------------------------------------------------------------------------------------------------------------------------------------------
"""


"""
----------------------------------------------------------------------------------------------------------------------------------------------------------
                                                       Part A. Functions for Generating Samples
In this part, we import the model for learning and create an object or instance from the imported class.
We define:
--- network_size:
    The topology of the deep neural network. For example, if it is [36, 100, 100, 100, 6], it means the deep neural network
    has one input layer with 36 neurons, three hidden layers each with 100 neurons, and an output layer with 6 neurons.
--- alpha:
    The learning rate for the set of weight matrix and slope multiplier.
--- epoch_of_learning:
    Learning epochs under which traditional SGD is performed in every epoch upon the set of weight matrix and slope multiplier.
--- Machine:
    The name of the object or instance created from the class "Brain".
                                            
----------------------------------------------------------------------------------------------------------------------------------------------------------
"""


def generate_state():

    state     = np.random.binomial(1, 0.5, size=9)
    state     = state.reshape((3, 3))

    return state


def return_state_value(state):

    state_value         = 0

    neighbor            = state.flatten() * np.array([1, 1, 1, 1, 0, 1, 1, 1, 1])
    number_of_neighbor  = neighbor.sum()

    self                = state.flatten() * np.array([0, 0, 0, 0, 1, 0, 0, 0, 0])
    number_of_self      = self.sum()

    if number_of_neighbor < 2:
        state_value     = np.ones(9) * ( 1 - number_of_self )

    if number_of_neighbor == 2:
        state_value     = np.ones(9)

    if number_of_neighbor == 3:
        state_value     = np.ones(9) * (     number_of_self )

    if number_of_neighbor > 3:
        state_value     = np.ones(9) * ( 1 - number_of_self )

    return state_value

# parameter referring to the batch size of the samples used to train the neural network. We recommend readers to try different batch size.
batch_size                = 1


"""
----------------------------------------------------------------------------------------------------------------------------------------------------------
                                                       Part B. Initializing Set of Weight Matrix and Importing Model
----------------------------------------------------------------------------------------------------------------------------------------------------------
"""


from Brain_for_learning import *
# this parameter refers to the topology of the neural network. We recommend readers to try different numbers.
network_size              = np.array([9, 100, 100, 100, 9])
# this parameter refers to intial slopes for the activation/sigmoid functions in the hidden and output layers of the neural network. We recommend readers to try different numbers.
slope                     = 30
# this parameter refers to learning rate. We recommend readers to try different numbers.
alpha                     = 0.000001
# this parameter refers to learning epochs. We recommend readers to try different numbers.
epoch_of_learning         = 50000000
# this parameter refers to the dropout rate in the learning phase. We recommend readers to try different numbers.
drop_rate                 = 0.015
# this parameter refers to the rate at which the momentum of the previous gradient affect the latter in the learning phase. We recommend readers to try different numbers.
momentum_rate             = 0.015

Machine                   = Brain(network_size, slope, alpha, epoch_of_learning, drop_rate, momentum_rate)


"""
----------------------------------------------------------------------------------------------------------------------------------------------------------
                                                       Part C. Generating Samples and Training by Model
----------------------------------------------------------------------------------------------------------------------------------------------------------
"""

# this parameter decides whether the program will train weight matrix upon existing weight matrix. We recommend readers to try different numbers.
retrain = False

if retrain == True:

    Machine.weight_list   = np.load("self.Conway_1_100x100x100_30_0.000001_20m_[1]_weight_list.npy" , allow_pickle=True)
    Machine.slope_list    = np.load("self.Conway_1_100x100x100_30_0.000001_20m_[1]_slope_list.npy"  , allow_pickle=True)


for i in range(epoch_of_learning):
    print(i)
    input_list  = list()
    output_list = list()
    for j in range(batch_size):

        state                       = generate_state()
        state_value                 = return_state_value(state)
        input_list .append(state.flatten())
        output_list.append(state_value)

    input_list  = np.asarray(input_list)
    output_list = np.asarray(output_list)
    Machine.learn_batch(input_list, output_list)


"""
----------------------------------------------------------------------------------------------------------------------------------------------------------
                                                       Part D. Saving the Trained Set of Weight Matrix for MWM-SGD
----------------------------------------------------------------------------------------------------------------------------------------------------------
"""

# these two lines save the trained set of weight matrix later to be used/selected in the dedcuing phase. We recommend readers to try different numbers.
np.save("self.Conway_1_100x100x100_30_0.000001_50m_[1]_weight_list"             , Machine.weight_list        )
np.save("self.Conway_1_100x100x100_30_0.000001_50m_[1]_slope_list"              , Machine.slope_list         )




