import random
import numpy as np
import sys
from time import time
import cPickle
import subprocess as sub
from glob import glob

from cppn.networks import CPPN
from cppn.softbot import Genotype, Phenotype, Population
from cppn.tools.algorithms import Optimizer
from cppn.tools.utilities import make_material_tree, natural_sort, count_neighbors
from cppn.objectives import ObjectiveDict
from cppn.tools.evaluation import evaluate_population
from cppn.tools.mutation import create_new_children_through_mutation
from cppn.tools.selection import pareto_selection


SEED = int(sys.argv[1])
random.seed(SEED)
np.random.seed(SEED)

sub.call("mkdir pickledPops{}".format(SEED), shell=True)
sub.call("mkdir data{}".format(SEED), shell=True)
sub.call("cp base.vxa data{}/".format(SEED), shell=True)

MAX_TIME = 1000  # [hours] Let's just set this to some really high number so it doesn't stop before 1000 generations.
GENS = 1001
POPSIZE = 8*6-1  # +1 for the randomly generated robot that is added each gen

IND_SIZE = (50, 50, 40)  # aka 100,000 voxels

CHECKPOINT_EVERY = 1  # gens

DIRECTORY = "."
start_time = time()


# def favor_appendages(displacement, ind):
#     total_vox = 0
#     total_neigh = 0
#     for name, details in ind.genotype.to_phenotype_mapping.items():
#         if name == "Data":
#             locations = details['state'] > 0
#             total_vox = np.sum(locations)
#             neigh = np.reshape(count_neighbors(details['state']), IND_SIZE)
#             total_neigh = np.sum(neigh[locations])
#     # print total_vox, float(total_neigh), displacement
#     return total_vox / float(total_neigh) * displacement


# def energy_efficiency(displacement, ind):
#     num_muscle_vox = 0
#     for name, details in ind.genotype.to_phenotype_mapping.items():
#         if name == "Data":
#             muscle = details['state'] > 2
#             num_muscle_vox = np.sum(num_muscle_vox)
#     return displacement / float(num_muscle_vox)


class MyGenotype(Genotype):
    """
    Defines a custom genotype, inheriting from base class Genotype.

    Each individual must have the following properties:

    The genotype consists of a single Compositional Pattern Producing Network (CPPN),
    with multiple inter-dependent outputs determining the material constituting each voxel
    (e.g. two types of active voxels, actuated in counter phase, and two passive voxel types, fat and bone)
    The material IDs in the phenotype mapping dependencies refer to a predefined palette of materials:
    (0: empty, 1: passiveSoft, 2: passiveHard, 3: active+, 4:active-)

    """
    def __init__(self):

        Genotype.__init__(self, orig_size_xyz=IND_SIZE)

        # self.add_network(CPPN(output_node_names=["Data"]))
        # self.to_phenotype_mapping.add_map(name="Data", tag="<Data>", func=half_and_half, output_type=int)

        self.add_network(CPPN(output_node_names=["shape", "muscleOrTissue", "muscleType", "tissueType"]))

        self.to_phenotype_mapping.add_map(name="Data", tag="<Data>", func=make_material_tree,
                                          dependency_order=["shape", "muscleOrTissue", "muscleType", "tissueType"],
                                          output_type=int)

        self.to_phenotype_mapping.add_output_dependency(name="shape", dependency_name=None, requirement=None,
                                                        material_if_true=None, material_if_false="0")

        self.to_phenotype_mapping.add_output_dependency(name="muscleOrTissue", dependency_name="shape",
                                                        requirement=True, material_if_true=None, material_if_false=None)

        self.to_phenotype_mapping.add_output_dependency(name="tissueType", dependency_name="muscleOrTissue",
                                                        requirement=False, material_if_true="1", material_if_false="2")

        self.to_phenotype_mapping.add_output_dependency(name="muscleType", dependency_name="muscleOrTissue",
                                                        requirement=True, material_if_true="3", material_if_false="4")


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

                # just make sure there is some material to simulate, even if all passive.
                if np.sum(state) == 0:
                    print "INVALID: Empty sim."
                    return False

        return True


# Now specify the objectives for the optimization.
# Creating an objectives dictionary
my_objective_dict = ObjectiveDict()

# Adding an objective named "fitness", which we want to maximize.
# This information is returned by Voxelyze in a fitness .xml file, with a tag named "distance"
my_objective_dict.add_objective(name="fitness", maximize=True, tag="<distance>")  # , meta_func=energy_efficiency)

# Add an objective to minimize the age of solutions: promotes diversity
my_objective_dict.add_objective(name="age", maximize=False, tag=None)

# quick test here to make sure evaluation is working properly:
# evaluate_population(my_pop)
# print [ind.fitness for ind in my_pop]

if len(glob("pickledPops/Gen_*.pickle")) == 0:
    # Initializing a population of SoftBots
    my_pop = Population(my_objective_dict, MyGenotype, MyPhenotype, pop_size=POPSIZE)

    # hack: see evaluation.py lines 63-77
    my_pop.material_wide_phase_offset = True

    my_pop.seed = SEED

    # Setting up our optimization
    my_optimization = Optimizer(my_pop, pareto_selection, create_new_children_through_mutation, evaluate_population)

else:
    successful_restart = False
    pickle_idx = 0
    while not successful_restart:
        try:
            pickled_pops = glob("pickledPops/*")
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

# finally, record the history of best robot at end of evolution so we can play it back in VoxCad
my_pop.individuals = [my_pop.individuals[0]]
evaluate_population(my_pop, record_history=True)

