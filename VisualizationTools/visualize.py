from __future__ import print_function

import copy
import warnings

import graphviz
import matplotlib.pyplot as plt
import numpy as np

import neat
from neat.six_util import iteritems


def plot_stats(statistics, ylog=False, view=False, filename='avg_fitness.svg'):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())

    plt.plot(generation, avg_fitness, 'b-', label="average")
    plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    plt.plot(generation, best_fitness, 'r-', label="best")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()


def plot_spikes(spikes, view=False, filename=None, title=None):
    """ Plots the trains for a single spiking neuron. """
    t_values = [t for t, I, v, u, f in spikes]
    v_values = [v for t, I, v, u, f in spikes]
    u_values = [u for t, I, v, u, f in spikes]
    I_values = [I for t, I, v, u, f in spikes]
    f_values = [f for t, I, v, u, f in spikes]

    fig = plt.figure()
    plt.subplot(4, 1, 1)
    plt.ylabel("Potential (mv)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, v_values, "g-")

    if title is None:
        plt.title("Izhikevich's spiking neuron model")
    else:
        plt.title("Izhikevich's spiking neuron model ({0!s})".format(title))

    plt.subplot(4, 1, 2)
    plt.ylabel("Fired")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, f_values, "r-")

    plt.subplot(4, 1, 3)
    plt.ylabel("Recovery (u)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, u_values, "r-")

    plt.subplot(4, 1, 4)
    plt.ylabel("Current (I)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, I_values, "r-o")

    if filename is not None:
        plt.savefig(filename)

    if view:
        plt.show()
        plt.close()
        fig = None

    return fig


def plot_species(statistics, view=False, filename='speciation.svg'):
    """ Visualizes speciation throughout evolution. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    species_sizes = statistics.get_species_sizes()
    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T

    fig, ax = plt.subplots()
    ax.stackplot(range(num_generations), *curves)

    plt.title("Speciation")
    plt.ylabel("Size per Species")
    plt.xlabel("Generations")

    plt.savefig(filename)

    if view:
        plt.show()

    plt.close()


def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
             node_colors=None, fmt='png'):
    """ Receives a genome and draws a neural network with arbitrary topology. """
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    # Create a directed graph to represent the network
    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    # Creates an ordered set
    inputs = set()
    # Loop through input nodes
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}
        # Add node to graph
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    # Loop through output nodes
    for k in config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue')}
        # Add node to graph
        dot.node(name, _attributes=node_attrs)

    # Exclude unused nodes from diagram
    if prune_unused:
        # Populate set of connections
        connections = set()
        for cg in genome.connections.values():
            if cg.enabled or show_disabled:
                connections.add((cg.in_node_id, cg.out_node_id))

        used_nodes = copy.copy(outputs)
        pending = copy.copy(outputs)
        while pending:
            new_pending = set()
            for a, b in connections:
                if b in pending and a not in used_nodes:
                    new_pending.add(a)
                    used_nodes.add(a)
            pending = new_pending
    # Include Unused nodes
    else:
        used_nodes = set(genome.nodes.keys())

    # Add nodes that aren't input/output to graph
    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        attrs = {'style': 'filled',
                 'fillcolor': node_colors.get(n, 'white')}
        dot.node(str(n), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            #if cg.input not in used_nodes or cg.output not in used_nodes:
            #    continue
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, view=view)

    return dot


def population_progression(config, stats, x_val, y_val):
    # Lists for Storing Data
    generation = []
    nodes = []
    connections = []
    fitness_vals = []
    accuracies = []
    
    # Go through each generation and collect data
    for i in range(len(stats.most_fit_genomes)):
        generation.append(i + 1)

        # Get Most Fit Genome
        fittest = stats.most_fit_genomes[i]

        # Find Num Nodes and Connections
        num_nodes = len(fittest.nodes)
        num_connections = len(fittest.connections)
        nodes.append(num_nodes)
        connections.append(num_connections)

        # Find Fitness
        fitness_vals.append(fittest.fitness)
        
        # Evaluate accuracy
        fittest_net = neat.nn.FeedForwardNetwork.create(fittest, config)
        correct_pos = 0
        correct_neg = 0
        for xi, xo in zip(x_val, y_val):
            output = fittest_net.activate(xi)
            if round(output[0]) == 1 and xo == 1:
                correct_pos += 1
            elif xo == 0:
                correct_neg += 1
        accuracy = 100*(correct_pos + correct_neg)/float(len(y_val))
        accuracies.append(accuracy)

    # Create Graphs
    # Graph 1
    plt.plot(generation, nodes, label="#Nodes")
    plt.plot(generation, connections, label="#Connections")

    plt.title("Progression of Popoulation (Structure)")
    plt.xlabel("Generations")
    plt.legend(loc="best")
    
    plt.savefig("pop_progression_structure.png")
    plt.show()
    plt.close()

    # Graph 2
    plt.plot(generation, fitness_vals, label="Fitness")

    plt.title("Progression of Popoulation (Fitness)")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.legend(loc="best")
    
    plt.savefig("pop_progression_fitness.png")
    plt.show()
    plt.close()

    # Graph 3
    plt.plot(generation, accuracies, label="Accuracy")

    plt.title("Progression of Popoulation (Accuracy)")
    plt.xlabel("Generations")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    
    plt.savefig("pop_progression_accuracy.png")
    plt.show()
    plt.close()


class ModifiedStatisticsReporter(neat.StatisticsReporter):
    def __init__(self):
        super().__init__()
        self.all_populations = []

    def post_evaluate(self, config, population, species, best_genome):
        # Store the population (all genome objects) of this generation
        # This is the added line
        self.all_populations.append(copy.deepcopy(population))

        # Store the best genome of this generation
        self.most_fit_genomes.append(copy.deepcopy(best_genome))

        # Store the fitnesses of the members of each currently active species.
        species_stats = {}
        for sid, s in species.species.items():
            species_stats[sid] = dict((k, v.fitness) for k, v in s.members.items())
        self.generation_statistics.append(species_stats)

    def display_vals(self):
        print("Num Populations: " + str(len(self.all_populations)))
        print("Num Most Fit Genomes: " + str(len(self.most_fit_genomes)))
        print("Num Generations: " + str(len(self.generation_statistics)))


class Species:
    def __init__(self, id, generations, fitness, accuracy):
        self.id = id
        self.generations = generations
        self.fitness = fitness
        self.accuracy = accuracy


def max_dict(dictionary):
    # Stores index of max val
    max = list(dictionary.keys())[0]
    for i in dictionary:
        if dictionary[i] > dictionary[max]:
            max = i
    return max


def loc_id(all_species, id):
    for i in range(len(all_species)):
        species = all_species[i]
        if species.id == id:
            return i
    return -1


def loc_tuple(all_genomes, id):
    for i in range(len(all_genomes)):
        genome = all_genomes[i]
        if genome[0] == id:
            return i


def find_accuracy(genome, x_val, y_val):
    correct_pos = 0
    correct_neg = 0
    for xi, xo in zip(x_val, y_val):
        output = genome.activate(xi)
        if round(output[0]) == 1 and xo == 1:
            correct_pos += 1
        elif round(output[0]) == 0 and xo == 0:
            correct_neg += 1
    accuracy = 100*(correct_pos + correct_neg)/float(len(y_val))
    return accuracy

def species_progression(config, stats, x_val, y_val):
    all_species = []
    i = 0

    for gen in stats.generation_statistics:
        # gen_genomes is a dictionary of all the genomes in the population at a specific generation
        gen_genomes = stats.all_populations[i]

        for species_id in gen:
            index = loc_id(all_species, species_id)

            # Species is new
            if index == -1:
                cur_gen = []
                cur_gen.append(i)

                cur_species = gen[species_id]
                cur_fitness = []
                max_genome_id = max_dict(cur_species)
                cur_fitness.append(cur_species[max_genome_id])
                
                cur_accuracy = []
                id_genome = gen_genomes[max_genome_id]
                net = neat.nn.FeedForwardNetwork.create(id_genome, config)
                cur_acc = find_accuracy(net, x_val, y_val)
                cur_accuracy.append(cur_acc)
                
                species = Species(species_id, cur_gen, cur_fitness, cur_accuracy)
                all_species.append(species)
            # Species already exists
            else:
                species = all_species[index]
                species.generations.append(i)

                cur_species = gen[species_id]
                max_genome_id = max_dict(cur_species)
                species.fitness.append(cur_species[max_genome_id])
                
                id_genome = gen_genomes[max_genome_id]
                net = neat.nn.FeedForwardNetwork.create(id_genome, config)
                cur_accuracy = find_accuracy(net, x_val, y_val)
                species.accuracy.append(cur_accuracy)
        i += 1

    # Graph Data
    for species in all_species:
        plt.plot(species.generations, species.fitness, label=str(species.id))
    
    plt.title("Progression of Different Species (Fitness)")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.legend(loc="best")

    plt.savefig("species_prog_fitness.png")
    plt.show()
    plt.close()
    
    for species in all_species:
        plt.plot(species.generations, species.accuracy, label=str(species.id))
    
    plt.title("Progression of Different Species (Accuracy)")
    plt.xlabel("Generations")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")

    plt.savefig("species_prog_accuracy.png")
    plt.show()
    plt.close()
