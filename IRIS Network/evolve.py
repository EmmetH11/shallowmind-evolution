from __future__ import print_function
import pandas as pd
import numpy as np
import os
import neat
import visualize


def get_vals(all_vals, amount):
    input = []
    output = []
    c1 = 0
    c2 = 0
    c3 = 0
    total = 0
    i = 0
    division = amount / 3

    while total < amount:
        if (all_vals.loc[i, "Species"] == 0 and c1 < division):
            input.append(all_vals.iloc[i, 0:4])
            output.append(0)
            c1 += 1
            total += 1
        elif (all_vals.loc[i, "Species"] == 1 and c2 < division):
            input.append(all_vals.iloc[i, 0:4])
            output.append(1)
            c2 += 1
            total += 1
        elif (all_vals.loc[i, "Species"] == 2 and c3 < division):
            input.append(all_vals.iloc[i, 0:4])
            output.append(2)
            c3 += 1
            total += 1
        i += 1
    
    final = []
    final.append(input)
    final.append(output)
    return final


# process values in train file (pandas)
dataframe = pd.read_csv("/Users/evanxie/Evan's Stuff/Shallow Mind/IRIS Network/Iris.csv")


# remove unnecessary column from dataframe
dataframe.drop("Id", axis = 1, inplace = True)


# convert classes to numbers
for i in range(dataframe.shape[0]):
    class_string = dataframe.loc[i, "Species"]
    if class_string == "Iris-setosa":
        dataframe.loc[i, "Species"] = 0
    elif class_string == "Iris-versicolor":
        dataframe.loc[i, "Species"] = 1
    else:
        dataframe.loc[i, "Species"] = 2


# normalize vals
for column in dataframe.columns:
    dataframe[column] = (dataframe[column] - dataframe[column].min()) / (dataframe[column].max() - dataframe[column].min())
for i in range(dataframe.shape[0]):
    dataframe.loc[i, "Species"] = int(dataframe.loc[i ,"Species"] * 2)


# randomly order rows
dataframe = dataframe.iloc[np.random.permutation(len(dataframe))]
dataframe.reset_index(inplace = True)
dataframe.drop("index", axis = 1, inplace = True)

# partition dataframe into training and test data
entire_df = dataframe.copy()
partitioned_vals = get_vals(dataframe, 120)

# create numpy arrays
input_vals = partitioned_vals[0]
output_vals = partitioned_vals[1]
input_tvals = entire_df.iloc[:, 0:4].values
output_tvals = entire_df.iloc[:, 4].values


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 120.0
        net = neat.nn.FeedForwardNetwork.create(genome, config) # create an actual network from the genome
        for i in range(len(input_vals)):
            expected_output = output_vals[i]
            input = input_vals[i]
            output = net.activate(input)

            # use distance to calculate fitness
            sum = 0
            for i in range(3):
                if i == expected_output:
                    sum += (1 - output[i]) ** 2
                else:
                    sum += output[i] ** 2

            genome.fitness -= (sum / 3) ** (1/2)


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    #p = neat.Population(config)
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-good')

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(20))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 10)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')

    num_suc = 0
    total = 0

    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for i in range(entire_df.shape[0]):
        output = winner_net.activate(input_tvals[i])
        
        print("Row " + str(i) + " Output: " + str(output))
        print("Correct Species: " + str(output_tvals[i]))
        print("\n")
        

        # find highest val node
        highest_index = 0
        for index in range(2):
            if output[index + 1] > output[highest_index]:
                highest_index = index + 1
        if highest_index == output_tvals[i]:
            num_suc += 1
        total += 1
    print("Successful: " + str(num_suc))
    print("Total: " + str(total))

    # Visualize
    node_names = {-1:'SL', -2:'SW', -3:'PL', -4:'PW', 0:'S', 1:'Ve', 2:'Vi'}
    visualize.draw_net(config, winner, True, node_names=node_names)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    run(config_path)
