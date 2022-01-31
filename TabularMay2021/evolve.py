from __future__ import print_function
import pandas as pd
import numpy as np
import os
import neat


# process values in train and test files (pandas)
dataframe = pd.read_csv("/Users/evanxie/Evan's Stuff/Shallow Mind/Network1/DataSet1/train.csv")
test_df = pd.read_csv("/Users/evanxie/Evan's Stuff/Shallow Mind/Network1/DataSet1/test.csv")

# remove unnecessary column from dataframe
dataframe.drop("id", axis = 1, inplace = True)
test_df.drop("id", axis = 1, inplace = True)

# randomly order rows
dataframe = dataframe.iloc[np.random.permutation(len(dataframe))]
dataframe.reset_index(inplace = True)
dataframe.drop("index", axis = 1, inplace = True)

# create numpy arrays
input_vals = dataframe.iloc[0:250, 0:50].values
output_vals = dataframe.iloc[0:250, 50].values
input_tvals = test_df.iloc[250:350, 0:50].values
output_tvals = dataframe.iloc[250:350, 50].values

# convert classes to numbers
for i in range(len(output_vals)):
    class_string = output_vals[i]
    if class_string == "Class_1":
        output_vals[i] = 0
    elif class_string == "Class_2":
        output_vals[i] = 1
    elif class_string == "Class_3":
        output_vals[i] = 2
    else:
        output_vals[i] = 3

for i in range(len(output_tvals)):
    class_string = output_tvals[i]
    if class_string == "Class_1":
        output_tvals[i] = 0
    elif class_string == "Class_2":
        output_tvals[i] = 1
    elif class_string == "Class_3":
        output_tvals[i] = 2
    else:
        output_tvals[i] = 3


def max_val(vals):
    highest_index = 0
    for i in range(len(vals)):
        if vals[i] > vals[highest_index]:
            highest_index = i
    return highest_index


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 250.0
        net = neat.nn.FeedForwardNetwork.create(genome, config) # create an actual network from the genome
        for i in range(len(input_vals)):
            expected_output = output_vals[i]
            input = input_vals[i]
            output = net.activate(input)

            # calculate fitness
            sum = 0
            for index in range(len(output)):
                if index == expected_output:
                    sum += (1 - output[index]) ** 2
                else:
                    sum += output[index] ** 2

            genome.fitness -= (sum ** (1/2) / 2)


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    #p = neat.Checkpointer.restore_checkpoint("/Users/evanxie/Evan's Stuff/Shallow Mind/Network1/neat-checkpoint-49")

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 50 generations.
    winner = p.run(eval_genomes, 50)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')

    classifications = []
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for i in range(len(input_tvals)):
        output = winner_net.activate(input_tvals[i])
        print(output)
        classifications.append(max_val(output))
    
    correct = 0
    total = 0
    for i in range(len(classifications)):
        if classifications[i] == output_tvals[i]:
            correct += 1
        total += 1
    print("Correct: " + correct)
    print("Total: " + total)

    print("Supposed Output:")
    print(output_tvals)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    run(config_path)
