"""
Single-pole balancing experiment using a continuous-time recurrent neural network (CTRNN).
"""

from __future__ import print_function

import multiprocessing
import os
import pickle

import cart_pole

import neat
from utils import visualize

import utils.filesaver as fs

runs_per_net = 5
simulation_seconds = 60.0
time_const = cart_pole.CartPole.time_step

def eval_genome(genome, config):
    net = neat.ctrnn.CTRNN.create(genome, config, time_const)

    fitnesses = []
    for runs in range(runs_per_net):
        sim = cart_pole.CartPole()
        net.reset() # reset because its recurrent

         # Run the given simulation for up to num_steps time steps.
        fitness = 0.0
        while sim.t < simulation_seconds:
            inputs = sim.get_scaled_state()
            action = net.advance(inputs, time_const, time_const)

             # Apply action to the simulated cart-pole
            force = cart_pole.discrete_actuator_force(action)
            sim.step(force)
            
            # Stop if the network fails to keep the cart within the position or angle limits.
            # The per-run fitness is the number of time steps the network can balance the pole
            # without exceeding these limits.
            if abs(sim.x) >= sim.position_limit or abs(sim.theta) >= sim.angle_limit_radians:
                break
            
            fitness = sim.t # fitness is the amount of time it lasts
    
    fitnesses.append(fitness)

    # The genome's fitness is its worst performance across all runs.
    return min(fitnesses)

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run(out_dir):
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-ctrnn')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(pe.evaluate)

    # Save the winner.
    with open(f'{out_dir}/winner-ctrnn', 'wb') as f:
        pickle.dump(winner, f)
    
    print(winner)

    visualize.plot_stats(stats, ylog=True, view=True, filename=f"{out_dir}/ctrnn-fitness.svg")
    visualize.plot_species(stats, view=True, filename=f"{out_dir}/ctrnn-speciation.svg")

    node_names = {-1: 'x', -2: 'dx', -3: 'theta', -4: 'dtheta', 0: 'control'}
    visualize.draw_net(config, winner, True, node_names=node_names, filename=f"{out_dir}/Digraph.gv")

    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename=f"{out_dir}/winner-ctrnn.gv")
    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename=f"{out_dir}/winner-ctrnn-enabled.gv", show_disabled=False)
    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename=f"{out_dir}/winner-ctrnn-enabled-pruned.gv", show_disabled=False, prune_unused=True)


if __name__ == '__main__':
    filesaver = fs.FileSaver("ctrnn-files", "rnn-exp", 3)
    out_dir = filesaver.getNextPath(createDir=True)
    run(out_dir)