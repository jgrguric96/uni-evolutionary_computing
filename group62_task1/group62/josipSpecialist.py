import sys, os
sys.path.insert(0, 'evoman')
from NEAT_GA.Node import Node
from NEAT_GA.Genes import Genes
from josipController import Controller as jControl
from statistics import mean
from NEAT_GA.Genome import Genome
from evoman.environment import Environment
from random import choices, choice, random, randrange, uniform
import numpy as np
import copy
import json

experiment_name = 'josipSpecialist'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Step One: Generate the initial population of individuals randomly. (First generation)
#
# Step Two: Repeat the following regenerational steps until termination:
#
#             Evaluate the fitness of each individual in the population (time limit, sufficient fitness achieved, etc.)
#             Select the fittest individuals for reproduction. (Parents)
#             Breed new individuals through crossover and mutation operations to give birth to offspring. #Look into Genes/Node code.
#             Replace the least-fit individuals of the population with new individuals. (again, Gene/node)
#
#             https://web.unbc.ca/~lucas0/papers/Old%20Papers/Efficient-Evolution-of-NN-Topologies.pdf
#             http://nn.cs.utexas.edu/downloads/papers/stanley.phd04.pdf
#             As well as Wiki of Evolutionary Algorithm

# initializes environment with ai player using random controller, playing against static enemy



# Todo: Apperently NEAT starts of with perceptrons (only input -> output) So i should start off with that as well.

c1, c2, c3 = 0.8, 0.8, 0.4
deltaT = 4.0
##use these
#stale_species = 15
MutateConnectionsChance = 0.25
PerturbChance = 0.90
#crossover_chance = 0.75
#LinkMutationChance = 2.0
#NodeMutationChance = 0.50
#BiasMutationChance = 0.40
#DisableMutationChance = 0.4
#EnableMutationChance = 0.2
#TimoutConstant = 20
step_size = 0.1
minimum_constant = 6.907755278982137
######
# champ of each species with more than 5 networks was copied
weight_mutation_chance = 0.8
#uniform_weight_mutation_chance = 0.9
interspecies_mating_rate = 0.001
new_node_mutation_chance = 0.03
#small_population_new_link_mutation = 0.05
large_population_new_link_mutation = 0.3
# the example paper used only sigmoidal transformation, while we use sigmoid and tanh

#Generations are also called population
GENERATIONS = 20

TRAINING_SESSIONS = 100

max_mean_fitness_per_generation = {}
best_representative = None
best_representative_fitness = {}
winning_representatives = {}

def gene_calculations(genome1, genome2):
    # TODO: Check this - This is organizing by the incorrect value. Should work now.
    #  It does work now when given a proper genome.
    try:
        genome1_connections = dict(sorted(genome1[0].genome.connections.items(), key=lambda item: item[1][0]))
        genome2_connections = dict(sorted(genome2[0].genome.connections.items(), key=lambda item: item[1][0]))
        # genome1_connections_reverse = dict(sorted(genome1[0].genome.connections.items(), key=lambda item: item[1][0], reverse=True))
    except TypeError:
        print("The following error occured: ", sys.exc_info()[0])
        print("Genome 1 connection items: ", genome1[0].genome.connections.items())
        print("Genome 2 connection items: ", genome2[0].genome.connections.items())
        print("Genome 1: ", genome1)
        print("Genome 2: ", genome2)
    excess, disjoint, w_difference, w_num = 0, 0, 0, 0
    g1_min = list(genome1_connections.items())[0][1][0]
    g1_max = list(genome1_connections.items())[-1][1][0]
    g2_min = list(genome2_connections.items())[0][1][0]
    g2_max = list(genome2_connections.items())[-1][1][0]
    # TODO: Check this. So far weight calculation didnt do average difference. Now it should be fixed
    #  Excess gene calculation is correct.
    #  Disjoint gene calculation is correct
    #  Same innovation calculation also works.
    for i in list(genome1_connections.items()):
        if genome2_connections.get(i[0]):
            w_difference += i[1][2].weight - genome2_connections.get(i[0])[2].weight
            w_num += 1
        else:
            if g2_min < i[1][0] < g2_max:
                disjoint += 1
            else:
                excess += 1
    # TODO: Check this
    for i in list(genome2_connections.items()):
        if genome1_connections.get(i[0]):
            continue
        else:
            if g1_min < i[1][0] < g1_max:
                disjoint += 1
            else:
                excess += 1
    w_num = 1 if w_num == 0 else w_num
    return excess, disjoint, w_difference/w_num

def speciation(generation, species_representatives):
    # Delta value = the compatibility distance between two genomes
    # E = excess genes;   D = disjoint genes;   W = average weight differences of matching genes
    # coefficients C# = They adjust the importance of the three factors and the factor N
    # Factor N = number of genes in the larger genome. Normalizes for genome size
    # WE can use base fitness value found in Evoman framework
    new_gen = []
    # TODO: Check this -
    #  Based on testing of definitions some issues were found that now seem fixed.
    #  Given this we can assume speciation now works fine given good generation and species_representatives values.
    for genome_1 in generation:
        current_species_delta = genome_1[0].species_compatibility
        old_species = genome_1[0].species
        genome_1[0].species = -1
        for genome_2 in species_representatives.values():
            # TODO: Gene calculations had issues. Still need to test for equal/disjoint
            excess_genes, disjoint_genes, w = gene_calculations(genome_1, genome_2)
            # TODO: Evaluate compatibility distance works fine - after all its just a formula
            delta = Genes.evaluate_compatibility_distance(excess_genes, disjoint_genes, w, c1, c2, c3, max(len(genome_1[0].genome.Genes), len(genome_2[0].genome.Genes)))
            if delta < deltaT:
                # TODO: Check this
                genome_1[0].species = copy.deepcopy(genome_2[0].species)
                genome_1[0].species_compatibility = delta
                break
        if genome_1[0].species == -1:
            # TODO: When genome1_species wasnt changed, this properly added a new species based on the last entry's species number + 1.
            genome_1[0].species = species_representatives.get(list(species_representatives.keys())[-1])[0].species + 1
            species_representatives[genome_1[0].species] = genome_1
        new_gen.extend([genome_1])
    return new_gen

def organize_by_species(population):
    # TODO: Check this. Does exactly as its suppsed to. One side effect I wasn't thinkg of is that the organization of species is not done in order of species, but rather in the order they appear in the population
    organized_by_species = {}
    for genome in population:
        if organized_by_species.get(genome[0].species):
            organized_by_species[genome[0].species] = organized_by_species.get(genome[0].species) + [[genome[0], genome[1]]]
        else:
            organized_by_species[genome[0].species] = [[genome[0], genome[1]]]
    return organized_by_species

def pair_selection(generation):
    # TODO: This isnt used
    population = []
    fitness = []
    weighted = []
    for genome in generation:
        population.extend([genome])
        fitness.extend([genome[1]])
    for i in range(len(fitness)):
        weighted.extend([(fitness[i]-min(fitness))/(max(fitness)-min(fitness))])
    while True:
        result = choices(population=population, weights=weighted, k=2)
        if result[0] != result[1]:
            break
    return result

def tournament_selection(species, p1 = 1.0, p2= 0.3, p3= 0.1):
    # TODO: Check this
    #Get 20% of species and take nearest power of 2.
    number_of_individuals = int((len(species)*0.4))
    p = random()
    if number_of_individuals == 2:
        first, second = choices(species, k=2)
        return first if first[1] >= second[1] else second
    if number_of_individuals <= 1:
        return choice(species)
    individuals = []
    first = second = third = None
    for i in range(number_of_individuals):
        individuals.extend((choices(species, k=2)))
    while len(individuals) > 3:
        ind_1 = randrange(0, len(individuals))
        ind_2 = randrange(0, len(individuals))
        if individuals[ind_1][1] > individuals[ind_2][1]:
            individuals.pop(ind_2)
        elif individuals[ind_1][1] < individuals[ind_2][1]:
            individuals.pop(ind_1)
        else:
            individuals.pop(choice([ind_1, ind_2]))
    individuals.sort(key=lambda individual: individual[1], reverse=True)
    first, second, third = individuals[0], individuals[1], individuals[2]
    return third if p < p3 else second if p < p2 else first



"""
 - The coefficients for measuring compatibility were c1 = 1.0, c2 = 1.0, and c3 = 0.4. 
 - In all experiments, δt = 3.0 (treshold for "difference" between species")
 - If the maximum fitness of a species did not improve in 15 generations, the networks in the stagnant species were
    not allowed to reproduce. 
 - The champion of each species with more than five networks was copied into the next generation unchanged. 
 - There was an 80% chance of a genome having its connection weights mutated, 
    in which case each weight had a 90% chance of being uniformly perturbed and a 
    10% chance of being assigned a new random value.
 - There was a 75% chance that an inherited gene was disabled if it was disabled in either parent. 
 - In each generation, 25% of offspring resulted from mutation without crossover. 
 - The interspecies mating rate was 0.001. 
 - In smaller populations, the probability of adding a new node was 0.03 and 
    the probability of a new link mutation was 0.05. In the larger population, 
    the probability of adding a new link was 0.3, because a larger population can tolerate a larger number of 
    prospective species and greater topological diversity. 
 - We used a modified sigmoidal transfer function, φ(x) = 1/(1+e−4.9x), at all nodes. 
    The steepened sigmoid allows more fine tuning at extreme activations. It is optimized to be close to 
    linear during its steepest ascent between activations −0.5 and 0.5. 
 - These parameter values were found experimentally: links need to be added significantly more often than nodes, 
    and an average weight difference of 3.0 is about as significant as one disjoint or excess gene. 
    Performance is robust to moderate variations in these values"""


def new_genome(species, species2=None, innovation_number=None, global_connection_list=None):
    # TODO: Perhaps genome_a and genome_b can be elected via tournaments
    # TODO: Check this. Tournament selection seems to work fine for values under 3 (still check for <=1 or >= 3. First run skipped over random chances. Next run i will force it
    #  NEW GENOME still has to be checked
    genome_a = tournament_selection(species)
    new_genome = None
    new_inov = [False, (0, 0), 0]
    new_global_conn_list = global_connection_list
    if random() < new_node_mutation_chance:
        # innov num = [False, (0, 0), 0]
        node_genes, inov_num_1, new_global_conn_list_1 = mutate_add_node(genome_a, innovation_number, global_connection_list)
        # TODO: Check if in the first we addressed the first node
        #  And in the second, the second node.
        #  Okay, i think it should work now. It does work now
        genome_a[0].genome.Nodes.extend([node_genes[0][0]])
        genome_a[0].genome.Nodes[node_genes[0][1].node_id] = node_genes[0][1]
        genome_a[0].genome.Genes.extend(node_genes[1])
        genome_a[0].genome.connections.get((node_genes[1][0].node_in.node_id, node_genes[1][1].node_out.node_id))[1] = False #This is unnecessary, just add [1] = False.
        for i in range(2):
            gene = node_genes[1][i]
            genome_a[0].genome.connections[(gene.node_in.node_id, gene.node_out.node_id)] = \
                [gene.innovation_number, gene.is_enabled, gene, gene.node_in, gene.node_out]
        new_genome = genome_a[0]
        new_inov[0] = inov_num_1[0]
        new_inov[2] += inov_num_1[2]
        innovation_number += inov_num_1[2]
        new_global_conn_list.update(new_global_conn_list_1)
        global_connection_list.update(new_global_conn_list_1)
        # return genome_a[0], new_inov, new_global_conn_list
    if random() < large_population_new_link_mutation:
        # TODO: This also seems to be fixed w.r.t. not adding connections
        node_gene, inov_num_2, new_global_conn_list_2 = mutate_add_connection(genome_a, innovation_number, global_connection_list)
        genome_a[0].genome.Genes.extend([node_gene[0]])
        genome_a[0].genome.Nodes[node_gene[1].node_id] = node_gene[1]
        genome_a[0].genome.connections[(node_gene[0].node_in.node_id, node_gene[0].node_out.node_id)] = [node_gene[0].innovation_number, node_gene[0].is_enabled, node_gene[0], node_gene[0].node_in, node_gene[0].node_out]
        new_genome = genome_a[0]
        new_inov[0] = inov_num_2[0]
        new_inov[2] += inov_num_2[2]
        innovation_number += inov_num_2[2]
        new_global_conn_list.update(new_global_conn_list_2)
        global_connection_list.update(new_global_conn_list_2)
        # return genome_a[0], new_inov, new_global_conn_list
    if new_genome is not None:
        if random() < MutateConnectionsChance:
            new_genome = mutate_weights(new_genome)
        return new_genome, new_inov, new_global_conn_list
    genome_b = tournament_selection(species)
    if species2:
        genome_b = tournament_selection(species2)
    is_first_gene_fitter = genome_a[1] > genome_b[1]
    is_second_gene_fitter = genome_a[1] < genome_b[1]
    if is_second_gene_fitter:
        new_genome = genome_b[0]
    elif is_first_gene_fitter:
        new_genome = genome_a[0]
    else:
        empty_genome = Genome(0, 0, None, global_connection_list)
        new_genome = jControl(0,empty_genome, innovation_number, global_connection_list)
        new_genome.species = genome_a[0].species
    first_vals = genome_a[0].genome.connections
    second_vals = genome_b[0].genome.connections
    # [self.innovation_number, self.Genes[-1].is_enabled, self.Genes[-1], node1, node2]
    # TODO: This should also take into account equal fitness vals. if equal it picks randomly
    # TODO: Double check above todo, as well as make sure there is randomness to disable/enable inheritance
    # TODO: Continue work on this. i was re-working how equal selection works. Perhaps this will make it work.
    for connection in new_genome.genome.connections.keys():
        if is_first_gene_fitter:
            if second_vals.get(connection):
                dominant = choice([genome_a[0], genome_b[0]])
                gene = dominant.genome.connections.get(connection)[2]
                if not (genome_a[0].genome.connections.get(connection)[2].is_enabled and genome_b[0].genome.connections.get(connection)[2].is_enabled):
                    if random() < 0.75:
                        gene.is_enabled = False
                try:
                    node1 = dominant.genome.Nodes[dominant.genome.connections.get(connection)[3].node_id]
                    node2 = dominant.genome.Nodes[dominant.genome.connections.get(connection)[4].node_id]
                    new_genome.genome.Nodes[node1.node_id] = node1
                    new_genome.genome.Nodes[node2.node_id] = node2
                except:
                    print("gene values: ", gene)
                    print("Node 1 values: ", node1)
                    print("Node 2 values: ", node2)
                    print("new_genome = ", new_genome)
                    print("new_genome_nodes = ", new_genome.genome.Nodes)
                for val in range(len(new_genome.genome.Genes)):
                    if new_genome.genome.Genes[val].node_in.node_id == gene.node_in.node_id and new_genome.genome.Genes[val].node_out.node_id == gene.node_out.node_id:
                        new_genome.genome.Genes[val] = gene
        elif is_second_gene_fitter:
            if first_vals.get(connection):
                dominant = choice([genome_a[0], genome_b[0]])
                gene = dominant.genome.connections.get(connection)[2]
                if not (genome_a[0].genome.connections.get(connection)[2].is_enabled and genome_b[0].genome.connections.get(connection)[2].is_enabled):
                    if random() < 0.75:
                        gene.is_enabled = False
                try:
                    node1 = dominant.genome.Nodes[dominant.genome.connections.get(connection)[3].node_id]
                    node2 = dominant.genome.Nodes[dominant.genome.connections.get(connection)[4].node_id]
                    new_genome.genome.Nodes[node1.node_id] = node1
                    new_genome.genome.Nodes[node2.node_id] = node2
                except:
                    print("gene values: ", gene)
                    print("Node 1 values: ", node1)
                    print("Node 2 values: ", node2)
                    print("new_genome = ", new_genome)
                    print("new_genome_nodes = ", new_genome.genome.Nodes)
                for val in range(len(new_genome.genome.Genes)):
                    if new_genome.genome.Genes[val].node_in.node_id == gene.node_in.node_id and new_genome.genome.Genes[val].node_out.node_id == gene.node_out.node_id:
                        new_genome.genome.Genes[val] = gene

    if not len(new_genome.genome.connections):
        for connection in genome_a[0].genome.connections.keys():
            if genome_b[0].genome.connections.get(connection):
                if random() >= 0.5:
                    try:
                        gene = genome_b[0].genome.connections.get(connection)[2]
                        node1 = genome_b[0].genome.Nodes[connection[0]]
                        node2 = genome_b[0].genome.Nodes[connection[1]]
                        conn = {}
                        conn[connection] = genome_b[0].genome.connections.get(connection)
                    except:
                        print("gene values: ", gene)
                        print("Node 1 values: ", node1)
                        print("Node 2 values: ", node2)
                        print("conn values: ", conn)
                        print("genome_b: ", genome_b)
                        print("genome_b.Nodes: ", genome_b[0].genome.Nodes)
                else:
                    try:
                        gene = genome_a[0].genome.connections.get(connection)[2]
                        node1 = genome_a[0].genome.Nodes[connection[0]]
                        node2 = genome_a[0].genome.Nodes[connection[1]]
                        conn = {}
                        conn[connection] = genome_a[0].genome.connections.get(connection)
                    except:
                        print("gene values: ", gene)
                        print("Node 1 values: ", node1)
                        print("Node 2 values: ", node2)
                        print("conn values: ", conn)
                        print("genome_a: ", genome_a)
                        print("genome_a.Nodes: ", genome_a[0].genome.Nodes)
                if not (genome_a[0].genome.connections.get(connection)[2].is_enabled
                        and genome_b[0].genome.connections.get(connection)[2].is_enabled):
                    if random() < 0.75:
                        gene.is_enabled = False
                if node1.node_id < len(new_genome.genome.Nodes):
                    new_genome.genome.Nodes[node1.node_id] = node1
                else:
                    new_genome.genome.Nodes.extend([node1])
                if node2.node_id < len(new_genome.genome.Nodes):
                    new_genome.genome.Nodes[node2.node_id] = node2
                else:
                    new_genome.genome.Nodes.extend([node2])
                if not new_genome.genome.connections.get(connection):
                    new_genome.genome.Genes.extend([gene])
                else:
                    for val in range(len(new_genome.genome.Genes)):
                        if new_genome.genome.Genes[val].node_in.node_id == gene.node_in.node_id and new_genome.genome.Genes[
                            val].node_out.node_id == gene.node_out.node_id:
                            new_genome.genome.Genes[val] = gene

                new_genome.genome.connections[connection] = [gene.innovation_number, gene.is_enabled, gene, gene.node_in, gene.node_out]
            else:
                if random() >= 0.5 or not len(new_genome.genome.connections):
                    try:
                        gene = genome_a[0].genome.connections.get(connection)[2]
                        node1 = genome_a[0].genome.Nodes[connection[0]]
                        node2 = genome_a[0].genome.Nodes[connection[1]]
                        conn = {}
                        conn[connection] = genome_a[0].genome.connections.get(connection)
                    except:
                        print("gene values: ", gene)
                        print("Node 1 values: ", node1)
                        print("Node 2 values: ", node2)
                        print("conn values: ", conn)
                        print("genome_a: ", genome_a)
                        print("genome_a.Nodes: ", genome_a[0].genome.Nodes)

                    if node1.node_id < len(new_genome.genome.Nodes):
                        new_genome.genome.Nodes[node1.node_id] = node1
                    else:
                        new_genome.genome.Nodes.extend([node1])
                    if node2.node_id < len(new_genome.genome.Nodes):
                        new_genome.genome.Nodes[node2.node_id] = node2
                    else:
                        new_genome.genome.Nodes.extend([node2])
                    if not new_genome.genome.connections.get(connection):
                        new_genome.genome.Genes.extend([gene])
                    else:
                        for val in range(len(new_genome.genome.Genes)):
                            if new_genome.genome.Genes[val].node_in.node_id == gene.node_in.node_id \
                                    and new_genome.genome.Genes[val].node_out.node_id == gene.node_out.node_id:
                                new_genome.genome.Genes[val] = gene

                    new_genome.genome.connections[connection] = conn.get(connection)

        for gen_b_conn in genome_b[0].genome.connections.keys():
            if not genome_a[0].genome.connections.get(gen_b_conn) and random() >= 0.5:
                try:
                    gene = genome_b[0].genome.connections.get(gen_b_conn)[2]
                    node1 = genome_b[0].genome.Nodes[gen_b_conn[0]]
                    node2 = genome_b[0].genome.Nodes[gen_b_conn[1]]
                    conn = {}
                    conn[connection] = genome_b[0].genome.connections.get(gen_b_conn)
                except:
                    print("gene values: ", gene)
                    print("Node 1 values: ", node1)
                    print("Node 2 values: ", node2)
                    print("conn values: ", conn)
                    print("genome_b: ", genome_b)
                    print("genome_b.Nodes: ", genome_b[0].genome.Nodes)

                if node1.node_id < len(new_genome.genome.Nodes):
                    new_genome.genome.Nodes[node1.node_id] = node1
                else:
                    new_genome.genome.Nodes.extend([node1])
                if node2.node_id < len(new_genome.genome.Nodes):
                    new_genome.genome.Nodes[node2.node_id] = node2
                else:
                    new_genome.genome.Nodes.extend([node2])
                if not new_genome.genome.connections.get(gen_b_conn):
                    new_genome.genome.Genes.extend([gene])
                else:
                    for val in range(len(new_genome.genome.Genes)):
                        if new_genome.genome.Genes[val].node_in.node_id == gene.node_in.node_id \
                                and new_genome.genome.Genes[val].node_out.node_id == gene.node_out.node_id:
                            new_genome.genome.Genes[val] = gene
                new_genome.genome.connections[gen_b_conn] = [gene.innovation_number, gene.is_enabled, gene, gene.node_in, gene.node_out]

    if random() < MutateConnectionsChance:
        new_genome = mutate_weights(new_genome)

    return new_genome, new_inov, new_global_conn_list

# def crossover(genome_a, genome_b, max_innovations):
#     first_vals = dict(sorted(genome_a[0].genome.connections.items(), key=lambda it:it[1]))
#     second_vals = dict(sorted(genome_b[0].genome.connections.items(), key=lambda it:it[1]))
#
#     new_genome = genome_selection_highest_fitness(first_vals,second_vals, genome_a, genome_b, is_first_gene_fitter=genome_a[1] > genome_b[1])
#     new_genome.global_innovation_number = max_innovations
#     return [new_genome]

def mutate_weights(new_genome):
    for gene in range(len(new_genome.genome.Genes)):
        if random() < weight_mutation_chance:
            pertrubed_by = random() * step_size * 2 - step_size
            if random() <= PerturbChance:
                new_genome.genome.Genes[gene].weight += pertrubed_by
                new_genome.genome.Genes[gene].weight = new_genome.genome.Genes[gene].weight
            else:
                new_genome.genome.Genes[gene].weight = random() * 4 - 2
        else:
            continue
    return new_genome

def crossover(generation: dict, avg_fit_per_species: dict, innovation_number:int, global_connection_list:dict, total_genomes = 100):
    total_fitness = 0
    new_generation = []
    #Get total fitness per species - This currently works fine
    for species_fit in avg_fit_per_species.keys():
        total_fitness += avg_fit_per_species.get(species_fit)
    # TODO: Cull 20% of species. Currently raised to 40 % - For the case of 10 species, chances are most of it will be eliminated with the min genome limit of 2. Percentage calculation works fine. Species removal works fine.
    for species in generation.keys():
        spec_len = len(generation.get(species))
        to_remove = int(spec_len*0.40)
        removable = generation.get(species)
        for i in range(to_remove):
            removable.pop(len(removable)-1)
        generation[species] = removable
    items = list(generation.keys())
    for key in items:
        if int(round((avg_fit_per_species.get(key)/total_fitness)*total_genomes)) < 2:
            generation.pop(key)

    for species in generation.keys():
        for i in range(int(round((avg_fit_per_species.get(species)/total_fitness)*total_genomes))):
            if i == 0 and len(generation.get(species)) >= 5:
                new_generation.extend([generation.get(species)[0][0]])
            else:
                species2 = None
                if random() < interspecies_mating_rate:
                    species2 = choice(list(generation.keys()))
                ngene, is_new_innov, global_connection_list = new_genome(copy.deepcopy(generation.get(species)), copy.deepcopy(generation.get(species2)), innovation_number, global_connection_list)
                if is_new_innov[0]:
                    innovation_number += is_new_innov[2]
                new_generation.extend([ngene])
    # TODO: The code below could've easily started an infinite loop.
    while len(new_generation) < total_genomes:
        missing_species = choice(list(generation.keys()))
        ngene, is_new_innov, global_connection_list = new_genome(copy.deepcopy(generation.get(missing_species)),
                                                                 copy.deepcopy(generation.get(missing_species)),
                                                                 innovation_number, global_connection_list)
        if is_new_innov[0]:
            innovation_number += is_new_innov[2]
        new_generation.extend([ngene])

    return new_generation, innovation_number, global_connection_list

#This can change weights
# def gene_mutation(species):
#     pass

#a single new connection gene is added connecting two previously unconnected nodes
#it has a random weight
def mutate_add_connection(species, inv_number, global_connection_list):
    while True:
        node1, node2 = choices(population=species[0].genome.Nodes, k=2)
        if node1 != node2:
            if not species[0].genome.connections.get((node1.node_id, node2.node_id)):
                if node1.layer is None and node2.layer is None:
                    if node1.node_id < 20 <= node2.node_id < 25:
                        break
                elif node1.layer is not None and node2.layer is not None:
                    if node1.layer != node2.layer:
                        if node1.layer < node2.layer:
                            break
                        else:
                            node_temp = node1
                            node1 = node2
                            node2 = node_temp
                            if not species[0].genome.connections.get((node1.node_id, node2.node_id)):
                                break
                else:
                    if node1.layer is None:
                        if node1.node_id < 20:
                            break
                        else:
                            node_temp = node1
                            node1 = node2
                            node2 = node_temp
                            if not species[0].genome.connections.get((node1.node_id, node2.node_id)):
                                break
                    else:
                        if 20 <= node2.node_id < 25:
                            break
                        else:
                            node_temp = node1
                            node1 = node2
                            node2 = node_temp
                            if not species[0].genome.connections.get((node1.node_id, node2.node_id)):
                                break

    is_new_innovation = [False, (0, 0), 0]
    if not global_connection_list.get((node1.node_id, node2.node_id)):
        is_new_innovation = [True, (node1.node_id, node2.node_id), 1]
        global_connection_list[(node1.node_id, node2.node_id)] = inv_number
    node2.gene_connections.extend([Genes(node_in=node1, node_out=node2, weight=random() * 4 - 2, is_enabled=True, innovation_number=inv_number)])
    return [node2.gene_connections[-1], node2], is_new_innovation, global_connection_list

#An existing connection is split, and the new node is placed where the old used to be
#The connection leading into the new node receives a weight of 1.
#The one leading out receives the same weight as the previous connection
def mutate_add_node(species, inv_number, global_connection_list):
    selected_gene = choice(species[0].genome.Genes)
    selected_gene.is_enabled = False
    node_genes = [[], []]
    layer = 10
    is_new_innovation = [False, (0, 0), 0]
    if selected_gene.node_in.node_type == "hidden":
        if selected_gene.node_out.node_type == "hidden":
            layer = mean([selected_gene.node_in.layer, selected_gene.node_out.layer])
        else:
            layer = selected_gene.node_in.layer + 10
    elif selected_gene.node_out.node_type == "hidden":
        layer = selected_gene.node_out.layer - 10
    node_genes[0].extend([Node(len(species[0].genome.Nodes), 0.1, "hidden", layer)])
    if not global_connection_list.get((selected_gene.node_in.node_id, node_genes[0][0].node_id)):
        global_connection_list[(selected_gene.node_in.node_id, node_genes[0][0].node_id)] = inv_number
        is_new_innovation[0] = True
        is_new_innovation[2] += 1
        inv_number += 1
    if not global_connection_list.get((node_genes[0][0].node_id, selected_gene.node_out.node_id)):
        global_connection_list[(node_genes[0][0].node_id, selected_gene.node_out.node_id)] = inv_number
        is_new_innovation[0] = True
        is_new_innovation[2] += 1
        inv_number += 1
    # TODO: The node that is added does not have a gene_connection value to it, thus it cannot calculate ANYTHING.
    #  In fact this will just disable nodes in the end, and why i didnt see any innovation.
    #  This is very likely happening in the "make connection" mutation
    node_genes[0][0].gene_connections.extend([Genes(selected_gene.node_in, node_genes[0][0], 1, True, global_connection_list.get((selected_gene.node_in.node_id, node_genes[0][0].node_id)))])
    selected_gene.node_out.gene_connections.extend([Genes(node_genes[0][0], selected_gene.node_out, selected_gene.weight, True, global_connection_list.get((node_genes[0][0].node_id, selected_gene.node_out.node_id)))])
    node_genes[1].extend([node_genes[0][0].gene_connections[-1]])
    node_genes[1].extend([selected_gene.node_out.gene_connections[-1]])
    node_genes[0].extend([selected_gene.node_out])
    # TODO: Because the connection to another node was changed, we also need to address the OUT node and change it as well.
    return node_genes, is_new_innovation, global_connection_list

start_controller = []
innov_number = 1
connection_list = {}
for i in range(TRAINING_SESSIONS):
    start_controller.append(jControl(0, global_innovation_number=innov_number, global_connection_list=connection_list))
    innov_number = start_controller[i].global_innovation_number

species_representatives = {}

# for enemy_number in range(1, 9):
gens = []
gens.extend([start_controller])
for gen_iteration in range(GENERATIONS):
    genome_fit = []
    max_mean_fitness_per_generation[gen_iteration] = [-1000, 0]
    # for iteration in range(len(gens[gen_iteration])):
    #     gens[gen_iteration][iteration].species = 1
    print("Starting with generation ", gen_iteration)
    for iteration in range(len(gens[gen_iteration])):
        env = Environment(experiment_name=experiment_name, speed="fastest", enemies=str(2), playermode="ai", player_controller=gens[gen_iteration][iteration], randomini='yes')
        #Returns the following: fitness, player.life, enemy.life, time
        data = env.play()
        if max_mean_fitness_per_generation.get(gen_iteration)[0] < data[0]:
            max_mean_fitness_per_generation[gen_iteration] = (data[0], max_mean_fitness_per_generation.get(gen_iteration)[1])
        max_mean_fitness_per_generation[gen_iteration] = (max_mean_fitness_per_generation.get(gen_iteration)[0],
                                                          max_mean_fitness_per_generation.get(gen_iteration)[1] + data[0])
        if best_representative is None:
            best_representative = [gens[gen_iteration][iteration], data[0], gen_iteration]
        elif best_representative[1] <= data[0] and best_representative[2] <= gen_iteration:
            best_representative = [gens[gen_iteration][iteration], data[0], gen_iteration]
        if not winning_representatives.get(gen_iteration):
            winning_representatives[gen_iteration] = best_representative[0]
        elif best_representative[1] <= data[0]:
            winning_representatives[gen_iteration] = best_representative[0]
        genome_fit.extend([[gens[gen_iteration][iteration], data[0]]])
        print(genome_fit[iteration][0].species)
        if not species_representatives.get(genome_fit[iteration][0].species):
            species_representatives[genome_fit[iteration][0].species] = copy.deepcopy(genome_fit[iteration])
        elif random() > 0.5:
            species_representatives[genome_fit[iteration][0].species] = copy.deepcopy(genome_fit[iteration])


    max_mean_fitness_per_generation[gen_iteration] = (max_mean_fitness_per_generation.get(gen_iteration)[0],
                                                      float(max_mean_fitness_per_generation.get(gen_iteration)[1]/len(gens[gen_iteration])))
    # Sorted seems to work fine
    gens[gen_iteration] = sorted(genome_fit,  key=lambda x: x[1], reverse=True)
    # Speciation seems to work fine given proper generation/ species representative values
    gens[gen_iteration] = speciation(gens[gen_iteration], species_representatives)
    #Organization of species also correctly done. Perhaps i could now re-structure it to also order by species number
    genome_per_species = organize_by_species(gens[gen_iteration])
    # Calculating adjusted fitness after speciation
    avg_fit = {}
    print("Species list: ", genome_per_species.keys())
    for species in genome_per_species.keys():
        sum_fit = 0
        for i in range(len(genome_per_species.get(species))):
            # genome_per_species[species][i][1] = genome_per_species[species][i][1]/len(genome_per_species.get(species))
            sum_fit += genome_per_species[species][i][1]
        avg_fit[species] = sum_fit/len(genome_per_species.get(species)) + minimum_constant
        print(f"Average fitness in species {species}: {avg_fit[species]-minimum_constant}")
        genome_per_species.get(species).sort(key=lambda i: i[1], reverse=True)
    next_generation, innov_number, connection_list = crossover(copy.deepcopy(genome_per_species), avg_fit,
                                                               innovation_number=innov_number,
                                                               global_connection_list=connection_list, total_genomes=TRAINING_SESSIONS)

    gens.extend([next_generation])

    # new_gen = gens[gen][:2]
    # for i in range(len(gens[gen])-2):
        # pairs = pair_selection(gens[gen])
        # new_gen.extend([crossover(pairs[0], pairs[1], innov_number), innov_number])

with open("josipSpecialist/ea_2_enemy6/ea_1_enemy6_max_mean_result_5.json", "w") as outfile:
    json.dump(max_mean_fitness_per_generation, outfile)

for final_run in range(5):
    final_test_env = Environment(experiment_name=experiment_name, speed="fastest", enemies=str(6), playermode="ai",
                      player_controller=best_representative[0], randomini='yes')
    # Returns the following: fitness, player.life, enemy.life, time
    end_data = final_test_env.play()
    best_representative_fitness[final_run+1] = end_data[0]

with open("josipSpecialist/ea_2_enemy6/ea_1_enemy6_best_species_result_5.json", "w") as outfile:
    json.dump(best_representative_fitness, outfile)

output = {"0": 0}
for i in range(1, len(winning_representatives.keys())):
    excss, dsjnt, w_diff = gene_calculations([winning_representatives.get(i-1)], [winning_representatives.get(i)])
    # TODO: Evaluate compatibility distance works fine - after all its just a formula
    output[i] = Genes.evaluate_compatibility_distance(excss, dsjnt, w_diff, c1, c2, c3,
                                                  max(len(winning_representatives.get(i-1).genome.Genes), len(winning_representatives.get(i).genome.Genes)))

with open("josipSpecialist/ea_2_enemy6/ea_1_enemy6_delta_difference_5.json", "w") as outfile:
    json.dump(output, outfile)
