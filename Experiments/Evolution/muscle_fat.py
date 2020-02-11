# from lxml import etree
import random
import numpy as np
import sys

from cppn.networks import CPPN
from cppn.softbot import Genotype, Phenotype, Population
from cppn.tools.algorithms import Optimizer
from cppn.tools.utilities import count_occurrences, make_material_tree, muscle_fat, one_muscle
from cppn.objectives import ObjectiveDict
from cppn.tools.evaluation import evaluate_population
from cppn.tools.mutation import create_new_children_through_mutation
from cppn.tools.selection import pareto_selection


SEED = 1  # int(sys.argv[1])
random.seed(SEED)
np.random.seed(SEED)

GENS = 0  # 1000
POPSIZE = 2  # 250  # +1 for the randomly generated robot that is added each gen

IND_SIZE = (50, 50, 40)  # (50, 50, 40) for 100,000...
MIN_PERCENT_FULL = 0.1
MIN_PERCENT_MUSCLE = 0.1
MAX_PERCENT_MUSCLE = 0.9

CHECKPOINT_EVERY = GENS+1  # ie. never  # GENS-1  # ie. last gen only

DIRECTORY = "."

# ROOT = etree.parse("assets/robot.data")


class MyGenotype(Genotype):
    """
    Defines a custom genotype, inheriting from base class Genotype.

    Each individual must have the following properties:

    The genotype consists of a Compositional Pattern Producing Network (CPPN),

    which dictates whether a voxel is present/absent and if present whether it is muscle or fat

    """
    def __init__(self):

        Genotype.__init__(self, orig_size_xyz=IND_SIZE)

        self.add_network(CPPN(output_node_names=["Data"]))

        self.to_phenotype_mapping.add_map(name="Data", tag="<Data>", func=muscle_fat, output_type=int)


class MyPhenotype(Phenotype):
    """
    Defines a custom phenotype, inheriting from the Phenotype class, which restricts the kind of robots that are valid

    """
    def is_valid(self):
        for name, details in self.genotype.to_phenotype_mapping.items():
            if np.isnan(details["state"]).any():
                return False

            if name == "Data":
                state = details["state"]

                if np.sum(state>0) < np.product(self.genotype.orig_size_xyz) * MIN_PERCENT_FULL:
                    return False

                if count_occurrences(state, [3, 4]) < np.product(self.genotype.orig_size_xyz) * MIN_PERCENT_MUSCLE:
                    return False

                if count_occurrences(state, [3, 4]) > np.product(self.genotype.orig_size_xyz) * MAX_PERCENT_MUSCLE:
                    return False
        return True


# Now specify the objectives for the optimization.
# Creating an objectives dictionary
my_objective_dict = ObjectiveDict()

# Adding an objective named "fitness", which we want to maximize.
# This information is returned by Voxelyze in a fitness .xml file, with a tag named "distance"
my_objective_dict.add_objective(name="fitness", maximize=True, tag="<distance>")

# Add an objective to minimize the age of solutions: promotes diversity
my_objective_dict.add_objective(name="age", maximize=False, tag=None)


# Initializing a population of SoftBots
my_pop = Population(my_objective_dict, MyGenotype, MyPhenotype, pop_size=POPSIZE)

# quick test here to make sure evaluation is working properly:
# evaluate_population(my_pop)
# print [ind.fitness for ind in my_pop]

# Setting up our optimization
my_optimization = Optimizer(my_pop, pareto_selection, create_new_children_through_mutation, evaluate_population)

my_optimization.run(my_pop, max_gens=GENS, checkpoint_every=CHECKPOINT_EVERY, directory=DIRECTORY)

# finally, record the history of best robot at end of evolution so we can play it back in VoxCad
my_pop.individuals = [my_pop.individuals[0]]
evaluate_population(my_pop, record_history=True)
# todo: this should be done every gen and there should be a script that grabs the best from a pickledPop and saves hist

