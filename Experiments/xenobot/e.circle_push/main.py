import random
import numpy as np
import sys
from time import time
import cPickle
import subprocess as sub
from glob import glob

from cppn.networks import CPPN, GeneralizedCPPN
from cppn.softbot import Genotype, Phenotype, Population
from cppn.tools.algorithms import Optimizer
from cppn.tools.utilities import natural_sort, make_one_shape_only, normalize
from cppn.objectives import ObjectiveDict
from cppn.tools.evaluation import evaluate_population
from cppn.tools.mutation import create_new_children_through_mutation
from cppn.tools.selection import pareto_selection

# inputs:
# 1 = seed
# 2 = num sims per gpu

SEED = int(sys.argv[1])
random.seed(SEED)
np.random.seed(SEED)

sub.call("mkdir pickledPops{}".format(SEED), shell=True)
sub.call("mkdir data{}".format(SEED), shell=True)
sub.call("cp base.vxa data{}/base.vxa".format(SEED), shell=True)

GENS = 5001
POPSIZE = int(sys.argv[2])#8*int(sys.argv[2])-1  # +1 for the randomly generated robot that is added each gen

CHECKPOINT_EVERY = 1#50  # gens
MAX_TIME = 47.7  # [hours] evolution does not stop; after MAX_TIME checkpointing occurs at every generation.

WORLD_SIZE = (12, 12, 7)
BODY_SIZE = (8, 8, 7)

DIRECTORY = "."
start_time = time()

# body space
body_x = np.zeros(WORLD_SIZE)
body_y = np.zeros(WORLD_SIZE)
body_z = np.zeros(WORLD_SIZE)

for x in range(BODY_SIZE[0]):
    for y in range(BODY_SIZE[1]):
        for z in range(BODY_SIZE[2]):
            body_x[x, y, z] = x
            body_y[x, y, z] = y
            body_z[x, y, z] = z

body_space_x = normalize(body_x)
body_space_y = normalize(body_y)
body_space_z = normalize(body_z)
body_space_r = normalize(np.power(np.power(body_space_x, 2) + np.power(body_space_y, 2) + np.power(body_space_z, 2), 0.5))
bias = np.ones(WORLD_SIZE)


def petri_dish(data):

    world = np.zeros(WORLD_SIZE, dtype=int)
    design = np.greater(data[:BODY_SIZE[0], :BODY_SIZE[1], :], 0)  # threshold based on Data cppn
    world[:BODY_SIZE[0], :BODY_SIZE[1], :] =  make_one_shape_only(design)

    world[10:12, 10:12, :2] = 2  # pellet

    world = np.swapaxes(world, 0,2)
    # world_flatten = world.reshape([z,-1])

    return world


def cilia_space(data):
    world = np.zeros(WORLD_SIZE, dtype=int)
    world[:BODY_SIZE[0], :BODY_SIZE[1], :] =  data[:BODY_SIZE[0], :BODY_SIZE[1], :]
    world = np.swapaxes(world, 0,2)
    return world


class MyGenotype(Genotype):
    """
    Defines a custom genotype, inheriting from base class Genotype.
    palette of materials: {0: empty, 1: cilia, 2: pellet}

    """
    def __init__(self):

        Genotype.__init__(self, orig_size_xyz=WORLD_SIZE)

        self.add_network(GeneralizedCPPN(output_node_names=["cilia_X", "cilia_Y"], 
                         bsx=body_space_x, bsy=body_space_y, bsz=body_space_z, bsd=body_space_r, b=bias))
        self.to_phenotype_mapping.add_map(name="cilia_X", tag="<cilia_X>", func=cilia_space, output_type=float)
        self.to_phenotype_mapping.add_map(name="cilia_Y", tag="<cilia_Y>", func=cilia_space, output_type=float)

        self.add_network(GeneralizedCPPN(output_node_names=["Data"], 
                         bsx=body_space_x, bsy=body_space_y, bsz=body_space_z, bsd=body_space_r, b=bias))
        self.to_phenotype_mapping.add_map(name="Data", tag="<Data>", func=petri_dish, output_type=int)


class MyPhenotype(Phenotype):
    """
    Defines a custom phenotype, inheriting from the Phenotype class, which restricts the kind of robots that are valid

    """
    def is_valid(self):
        for name, details in self.genotype.to_phenotype_mapping.items():
            if np.isnan(details["state"]).any():
                print "INVALID: Nans in phenotype."
                return False

            if name == "Data":
                state = details["state"]

                # make sure there is some material to simulate
                if np.sum(state==1) < 16:
                    print "INVALID: Empty sim."
                    return False

        return True


# Now specify the objectives for the optimization.
# Creating an objectives dictionary
my_objective_dict = ObjectiveDict()

# Adding an objective named "fitness", which we want to maximize.
# This information is returned by Voxelyze in a fitness .xml file, with a tag named "fitness_score"
my_objective_dict.add_objective(name="fitness", maximize=True, tag="<fitness_score>")

# Add an objective to minimize the age of solutions: promotes diversity
my_objective_dict.add_objective(name="age", maximize=False, tag=None)

# quick test here to make sure evaluation is working properly:
# my_pop = Population(my_objective_dict, MyGenotype, MyPhenotype, pop_size=POPSIZE)
# my_pop.seed = SEED
# my_pop.individuals = [my_pop.individuals[0]]
# evaluate_population(my_pop, record_history=True)
# evaluate_population(my_pop)
# print [ind.fitness for ind in my_pop]

if len(glob("pickledPops{}/Gen_*.pickle".format(SEED))) == 0:
    # Initializing a population of SoftBots
    my_pop = Population(my_objective_dict, MyGenotype, MyPhenotype, pop_size=POPSIZE)

    my_pop.seed = SEED

    # Setting up our optimization
    my_optimization = Optimizer(my_pop, pareto_selection, create_new_children_through_mutation, evaluate_population)

else:
    successful_restart = False
    pickle_idx = 0
    while not successful_restart:
        try:
            pickled_pops = glob("pickledPops{}/*".format(SEED))
            last_gen = natural_sort(pickled_pops, reverse=True)[pickle_idx]
            with open(last_gen, 'rb') as handle:
                [optimizer, random_state, numpy_random_state] = cPickle.load(handle)
            successful_restart = True

            my_pop = optimizer.pop
            my_optimization = optimizer
            my_optimization.continued_from_checkpoint = True
            my_optimization.start_time = time()

            random.setstate(random_state)
            np.random.set_state(numpy_random_state)

            print "Starting from pickled checkpoint: generation {}".format(my_pop.gen)

        except EOFError:
            # something went wrong writing the checkpoint : use previous checkpoint and redo last generation
            sub.call("touch IO_ERROR_$(date +%F_%R)", shell=True)
            pickle_idx += 1
            pass


my_optimization.run(max_hours_runtime=MAX_TIME, max_gens=GENS, checkpoint_every=CHECKPOINT_EVERY, directory=DIRECTORY)


# print "That took a total of {} minutes".format((time()-start_time)/60.)
# # finally, record the history of best robot at end of evolution so we can play it back in VoxCad
# my_pop.individuals = [my_pop.individuals[0]]
# evaluate_population(my_pop, record_history=True)



