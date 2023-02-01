from NEAT_GA import Node, Genes
import random

def create_starter_nodes():
    node_list = []
    for x in range(20):
        node_list.extend([Node.Node(x, 0.1, "sensor")])
    for x in range(5):
        node_list.extend([Node.Node(x+20, 0.1, "output")])
    return node_list

class Genome:
    def __init__(self, node_num=0, gene_num=1, parent_genome=None, innovation_number=None, global_conn_list = None):
        #The code itself starts with 25 nodes. 20 Sensors, 5 output.
        self.connections = {}
        self.innovation_number = innovation_number
        self.global_connection_list = global_conn_list
        if not parent_genome:
            self.Nodes = create_starter_nodes()
            for i in range(node_num):
                self.create_new_node()
            self.Genes = []
            self.create_random_genes(gene_num)
            #self.layers = 3
            self.species = 1
        else:
            self.Nodes = parent_genome[0]
            self.Genes = parent_genome[1]
            self.connections = parent_genome[2]
            #lets skip layers for now and make an assumption the hidden layers are just that.
            #self.layers = parent_genome.layers

    #Creates a new node
    def create_new_node(self):
        self.Nodes.extend([Node.Node(self.Nodes.__len__(), 0.1, "hidden", 10)])

    def create_new_gene(self, node1, node2, is_global):
        if not is_global:
            self.Genes.extend([Genes.Genes(node1, node2, random.random()*random.choice([-2, 2]), True, self.innovation_number)])
            self.Nodes[node2.node_id].gene_connections.extend([self.Genes[-1]])
            self.connections[(node1.node_id, node2.node_id)] = [self.innovation_number, self.Genes[-1].is_enabled, self.Genes[-1], node1, node2]
            self.global_connection_list[(node1.node_id, node2.node_id)] = self.innovation_number
            self.innovation_number += 1
        else:
            self.Genes.extend([Genes.Genes(node1, node2, random.random()*random.choice([-2, 2]), True, self.global_connection_list.get((node1.node_id, node2.node_id)))])
            self.Nodes[node2.node_id].gene_connections.extend([self.Genes[-1]])
            self.connections[(node1.node_id, node2.node_id)] = [self.global_connection_list.get((node1.node_id, node2.node_id)), self.Genes[-1].is_enabled, self.Genes[-1], node1, node2]



    #Creates random connection genes
    def create_random_genes(self, gene_num):
        for x in range(gene_num):
            #Select two genes for random connection as long as they arent of the same type
            while True:
                node1 = random.choice(self.Nodes)
                node2 = random.choice(self.Nodes)
                # node2 = self.Nodes[25]
                if not self.connections.get((node1.node_id, node2.node_id)):
                    if node1.node_type != "output" and node2.node_type != "sensor" and node1.node_type != node2.node_type:
                        if node2.node_type == "hidden":
                            node3 = node2
                            node4 = random.choice(self.Nodes[20:25])
                        break

            is_global_12 = self.global_connection_list.get((node1.node_id, node2.node_id))
            self.create_new_gene(node1, node2, is_global_12)
            if node2.node_type == "hidden":
                is_global_34 = self.global_connection_list.get((node3.node_id, node4.node_id))
                self.create_new_gene(node3, node4, is_global_34)
            # self.Genes.extend([Genes.Genes(node1, node2, 1, True, self.innovation_number)])
            # self.Nodes[node2.node_id].gene_connections.extend([self.Genes[-1]])
            # self.connections[(node1.node_id, node2.node_id)] = [self.innovation_number, self.Genes[-1].is_enabled]
            # if not is_global_12:
            #     self.global_connection_list[(node1.node_id, node2.node_id)] = self.innovation_number
            #     self.innovation_number += 1
            # if node2.node_type == "hidden":
            #     if not self.connections.get((node3.node_id, node4.node_id)):
            #         self.Genes.extend([Genes.Genes(node3, node4, 1, True, self.innovation_number)])
            #         self.Nodes[node4.node_id].gene_connections.extend([self.Genes[-1]])
            #         self.connections[(node3.node_id, node4.node_id)] = [self.innovation_number, self.Genes[-1].is_enabled]
            #         if not is_global_34:
            #             self.global_connection_list[(node3.node_id, node4.node_id)] = self.innovation_number
            #             self.innovation_number += 1

            # setattr(self.Nodes[node1.node_id], 'connections', self.Nodes[node1.node_id].connections + [self.Genes[-1].innovation_number+1])

