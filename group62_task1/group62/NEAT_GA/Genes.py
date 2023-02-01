class Genes:
    def __init__(self, node_in=None, node_out=None, weight=None, is_enabled=None, innovation_number=None):
        self.node_in = node_in
        self.node_out = node_out
        self.weight = weight
        self.is_enabled = is_enabled
        self.innovation_number = innovation_number

    @staticmethod
    def evaluate_compatibility_distance(e:int, d:int, w:float, c1, c2, c3, n):
        # Delta value = the compatibility distance between two genomes
        # E = excess genes;   D = disjoint genes;   W = average weight differences of matching genes
        # coefficients C# = They adjust the importance of the three factors and the factor N
        # Factor N = number of genes in the larger genome. Reduced to 1 if N is below specific number
        # We can use base fitness value found in Evoman framework
        # TODO: Check if 20 is optimal.
        if n < 20:
            n = 1
        delta_value = (c1*e)/n + (c2*d)/n + c3*w
        return delta_value

    @staticmethod
    def average_adjusted_fitness(fitness, neighbour_genomes):
        # Formula: N'_j = (Sigma[i=1, Nj](f_ij))/f
        # Adjusted fitness: division of original fitness with the total number of individuals in the same species
        # n_j = number of individuals in species j
        # nNew_j = new off spring replacing the entire population of the species
        # j = species j
        # j[i] = adjusted fitness of individual i in species j
        # f is the mean adjusted fitness of the entire population
        return fitness/neighbour_genomes

    def gene_mutation(self):
        pass

    def gene_add_gene(self):
        pass
