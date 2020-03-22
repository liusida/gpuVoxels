# This file contains high level logic.

# psuedocode of this file
#
# if last generation of population exist
#   read them in
# else
#   randomly generate a population
# send the population to simulation
# read the output report
# select high fitness population
# use report to train a model
# use the model to mutate high fitness population
# write the next generation of population with high fitnesses and mutants

import voxelyze as vx
import numpy as np
import shutil
import random
random.seed(1)
np.random.seed(1)
experiment_name = "v0.6"
population_size = 10
generation = 0
body_dimension = (3, 3, 3)

vx.clear_workspace()

# try to resume from last experiment
population, generation = vx.load_last_generation(experiment_name)
# if failed, start from scratch
if population is None:
    generation = 0
    population = {"genotype": [], "body": [], "phaseoffset": [],
                  "firstname": [], "lastname": []}
    # random initialization
    for robot_id in range(population_size):
        body_random = np.random.random(body_dimension)
        body = np.zeros_like(body_random, dtype=int)
        body[body_random < 0.5] = 1
        body = vx.largest_component(body)
        phaseoffset = np.random.random(body_dimension)
        population["genotype"].append("010101")
        population["body"].append(body)
        population["phaseoffset"].append(phaseoffset)
        population["firstname"].append(vx.names.get_first_name())
        population["lastname"].append(vx.names.get_last_name())

# infinity evolutionary loop
while(True):
    # write vxa vxd
    foldername = vx.prepare_directories(experiment_name, generation)
    vx.copy_vxa(experiment_name, generation)
    vx.write_all_vxd(experiment_name, generation, population)

    # start simulator
    vx.start_simulator(experiment_name, generation)

    # record a brief history for the bestfit
    vx.record_bestfit_history(experiment_name, generation)

    # read reporter
    sorted_result = vx.read_report(experiment_name, generation)

    # select the first half
    selected = vx.empty_population_like(population)
    half = int(len(sorted_result["id"])/2)
    for i in sorted_result["id"][:half]:
        for key in population.keys():
            selected[key].append(population[key][i])

    # mutate first half to two mutant groups
    mutation = vx.mutation.Mutation()
    mutant1 = mutation.mutate(selected)
    mutant2 = mutation.mutate(selected)

    # combine two mutant groups into next generation
    next_generation = vx.empty_population_like(population)
    vx.add_population(mutant1, next_generation)
    vx.add_population(mutant2, next_generation)

    # report the fitness
    msg = f"Simulation for generation {generation} finished.\nThe top 3 bestfit fitness score of this generation are \n"
    for i in range(3):
        msg += f"{population['firstname'][sorted_result['id'][i]]} {population['lastname'][sorted_result['id'][i]]}'s fitness score: {sorted_result['fitness'][i]:.1e} \n"
    print(msg)

    # next generation
    generation += 1
    population = next_generation
