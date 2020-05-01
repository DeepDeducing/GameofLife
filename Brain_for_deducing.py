import numpy as np
from scipy.special import expit


class Brain(object):
    def __init__(self, network_size, beta, epoch_of_deducing, drop_rate):

        self.network_size                 = network_size
        self.number_of_layers             = self.network_size.shape[0]

        self.beta                         = beta
        self.epoch_of_deducing            = epoch_of_deducing

        self.drop_rate                    = drop_rate


    def activator(self, x):
        return expit(x)


    def activator_output_to_derivative(self, output):
        return output * ( 1 - output)


    def generate_values_for_each_layer(self, input):

        layer_list                = list()

        layer                     = input

        layer_list.append(layer)

        for i in range(self.number_of_layers - 2):


            binomial              = np.atleast_2d(np.random.binomial(1, 1 - self.drop_rate, size=self.network_size[1 + i]))

            layer                 = self.activator(np.dot(layer_list[-1]                          , self.weight_list[i]                                                          ) * self.slope_list[i] )

            layer                *= binomial

            layer_list.append(layer)

        layer = self.activator(np.dot(layer_list[-1], self.weight_list[-1]) * self.slope_list[-1])

        layer_list.append(layer)

        return   layer_list


    def train_for_input_inner(self,
                       layer_list, desired_output):

        layer_final_error      = desired_output - layer_list[-1]

        layer_delta            = layer_final_error                                                                                              * self.activator_output_to_derivative(layer_list[-1])           * self.slope_list[-1]

        for i in range(self.number_of_layers - 2):

            layer_delta        = (layer_delta.dot( self.weight_list[- 1 - i].T                                                          ) )     * self.activator_output_to_derivative(layer_list[- 1 - 1 - i])  * self.slope_list[-1 -1 -i]

        layer_delta        = (layer_delta.dot( self.weight_list[0].T                                                                    ) )     * self.activator_output_to_derivative(layer_list[0])

        self.missing_input_inner  += layer_delta  * self.beta    * self.missing_input_resistor


    def deduce_batch(self, missing_input_inner, missing_input_resistor, desired_output, weight_list, slope_list):


        self.weight_list = weight_list
        self.slope_list  = slope_list


        self.missing_input_inner    = missing_input_inner
        self.missing_input_resistor = missing_input_resistor


        layer_list = self.generate_values_for_each_layer(self.activator( self.missing_input_inner ))
        self.train_for_input_inner(layer_list, desired_output)


        return self.missing_input_inner


