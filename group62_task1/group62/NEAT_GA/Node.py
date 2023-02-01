# I am very likely misunderstanding the naming conventions of Evolutionary Algorithms.
# I believe it best to first research naming, then follow up by implementation of NEAT upon GA learning.
# Look into josipController for a general algorithm flow.
class Node:
    def __init__(self, node_id, bias, node_type, layer = None):
        self.node_id = node_id
        self.bias = bias
        # activation type could be ignored, but lets try sigmoid and tanh Activation types are: sigmoid, tanh, linear, etc
        self.node_type = node_type
        if node_type == "hidden":
            self.activation_type = "tanh"
        else:
            self.activation_type = "sigmoid"
        self.node_summation = 0
        self.gene_connections = []
        self.layer = layer



