from __future__ import print_function
import os
from turtle import pos
import neat
#import visualize
import pandas as pd
import numpy as np
import preprocess

os.environ["PATH"] += os.pathsep + "C:\Program Files\Graphviz\\bin"

dataset = preprocess.load_data()
x_train, y_train, x_val, y_val = preprocess.prepare_data(dataset)
DATASET_SIZE = len(x_val)
GENERATIONS = 500

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = DATASET_SIZE
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(x_train, y_train):
            output = net.activate(xi)
            genome.fitness -= (output[0] - xo) ** 2

def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to x generations.
    winner = p.run(eval_genomes, GENERATIONS)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    correct_pos = 0
    correct_neg = 0
    pos_guesses = 0
    for xi, xo in zip(x_val, y_val):
        output = winner_net.activate(xi)
        #print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))
        print("Expected output {!r}, got {!r}".format(xo, output))
        if round(output[0]) == 1:
            pos_guesses += 1
            if xo == 1:
                correct_pos += 1
        elif xo == 0:
            correct_neg += 1
    print("\n----- VALIDATION RESULTS -----\nACCURACY:", (correct_pos + correct_neg)/float(len(y_val)))
    print("PRECISION:", correct_pos/float(sum(y_val)))
    print("RECALL:", correct_pos/float(pos_guesses))

    node_names = dict(zip(dataset.keys(), range(-1, -1*len(dataset.keys()), -1)))
    node_names[0] = "TARGET"
    # visualize.draw_net(config, winner, True, node_names=node_names)
    # visualize.plot_stats(stats, ylog=False, view=True)
    # visualize.plot_species(stats, view=True)

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 10)

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    run(config_path)