from evoman.controller import Controller as Cnt
from random import choices
from NEAT_GA import Node, Genes, Genome
import numpy as np

def random_output():
    return choices([0,1], k=5)

#I use sigmoid activation for Outputs
def sigmoid_activation(x):
	return 1./(1.+np.exp(-x))

#I use tanh for hidden layers.
def tanh_activation(x):
    return 2./(1.+np.exp(-2*x))-1


class Controller(Cnt):

    """ Step One: Generate the initial population of individuals randomly. (First generation)

        Step Two: Repeat the following regenerational steps until termination:

            Evaluate the fitness of each individual in the population (time limit, sufficient fitness achieved, etc.)
            Select the fittest individuals for reproduction. (Parents)
            Breed new individuals through crossover and mutation operations to give birth to offspring. #Look into Genes/Node code.
            Replace the least-fit individuals of the population with new individuals. (again, Gene/node)

            https://web.unbc.ca/~lucas0/papers/Old%20Papers/Efficient-Evolution-of-NN-Topologies.pdf
            http://nn.cs.utexas.edu/downloads/papers/stanley.phd04.pdf
            As well as Wiki of Evolutionary Algorithm"""

    def __init__(self, hidden_layers, genome=None, global_innovation_number=0, global_connection_list = None):
        self.hidden_layers = hidden_layers
        self.all_output = []
        self.genome = genome
        self.species = 1
        self.species_compatibility = None
        self.global_innovation_number = global_innovation_number
        self.global_connection_list = global_connection_list
        if not self.genome:
            self.genome = Genome.Genome(innovation_number=self.global_innovation_number, global_conn_list = self.global_connection_list)
            self.global_innovation_number = self.genome.innovation_number
            self.global_connection_list.update(self.genome.global_connection_list)

    # This isnt working well. It could be edge cases, it could be faulty logic.
    # Whatever the case is: after a while it reaches max recurssion depth.
    def summation_activation(self, node):
        """
        if current node has no connections:
            return sigmoid_activation(node_value)
        output = 0
        for connection in current node:
            output += summation_activation(node_id) * gene_weight
        return sigmoid_activation(output + node_bias)"""

        if len(node.gene_connections) == 0:
            return sigmoid_activation(node.node_summation)
        output = 0
        for connection in node.gene_connections:
            if connection.is_enabled:
                output += self.summation_activation(connection.node_in) * connection.weight
        return sigmoid_activation(output + node.bias)

    #This one works fine
    def summation_activation_nonrecurrsion(self):
        output = [0, 0, 0, 0, 0]
        for i in range(25,len(self.genome.Nodes)):
            summation = 0
            for connection in self.genome.Nodes[i].gene_connections:
                summation += connection.weight * connection.node_in.node_summation
            self.genome.Nodes[i].node_summation = tanh_activation(summation)
        for i in range(20,25):
            summation = 0
            for connection in self.genome.Nodes[i].gene_connections:
                summation += connection.weight * connection.node_in.node_summation
            if self.genome.Nodes[i].gene_connections == 0:
                output[i-20] = 0
            else:
                output[i-20] = sigmoid_activation(summation)

        return output

    def generate_output(self, inputs):
        for i in range(20):
            self.genome.Nodes[i].node_summation = inputs[i]
        output = []

        #I want a dot calculation of input*weights + bias
        # for i in range(5):
        #     if len(self.genome.Nodes[i+20].gene_connections) == 0:
        #         output.extend([0])
        #     else:
        #         output.extend([self.summation_activation(self.genome.Nodes[i+20])])
        for i in range(5):
            if len(self.genome.Nodes[i+20].gene_connections) == 0:
                output.extend([0])
            else:
                output.extend([self.summation_activation(self.genome.Nodes[i+20])])

        # output = self.summation_activation_nonrecurrsion()

        return output

    # I cannot define the fitness of each action per frame. I can only get the fitness value at the end of a training session.
    # Meaning i should let it train for multiple generations doing a single random action. Maybe just pure random actions.
    def control(self, params, cont):
        # if self.hidden_layers == 0:
        #     self.create_bases_nn(params)
        # Normalization against the maximum value.
        # inputs = (params - min(params)) / float((max(params) - min(params)))
        inputs = params
        output = self.generate_output(inputs)

        # takes decisions about sprite actions
        if output[0] > 0.5:
            left = 1
        else:
            left = 0

        if output[1] > 0.5:
            right = 1
        else:
            right = 0

        if output[2] > 0.5:
            jump = 1
        else:
            jump = 0

        if output[3] > 0.5:
            shoot = 1
        else:
            shoot = 0

        if output[4] > 0.5:
            release = 1
        else:
            release = 0

        self.all_output.extend(output)

        return [left, right, jump, shoot, release]
