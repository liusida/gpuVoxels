import hashlib
from lxml import etree
import subprocess as sub
import numpy as np


def evaluate_population(pop, record_history=False):

    seed = pop.seed

    # clear old .vxd robot files from the data directory
    sub.call("rm data{}/*.vxd".format(seed), shell=True)

    # remove old sim output.xml if we are saving new stats
    if not record_history:
        sub.call("rm output{}.xml".format(seed), shell=True)

    num_evaluated_this_gen = 0

    for n, ind in enumerate(pop):

        # don't evaluate if invalid
        if not ind.phenotype.is_valid():
            for rank, goal in pop.objective_dict.items():
                if goal["name"] != "age":
                    setattr(ind, goal["name"], goal["worst_value"])

            print "Skipping invalid individual"

        # otherwise create a vxd
        else:
            num_evaluated_this_gen += 1
            pop.total_evaluations += 1

            (x, y, z) = ind.genotype.orig_size_xyz

            root = etree.Element("VXD")  # new vxd root

            if record_history:
                sub.call("rm champ_{0}_gen{1}.hist".format(seed, pop.gen), shell=True)
                history = etree.SubElement(root, "RecordHistory")
                history.set('replace', 'VXA.Simulator.RecordHistory')
                etree.SubElement(history, "RecordStepSize").text = '100'

            structure = etree.SubElement(root, "Structure")
            structure.set('replace', 'VXA.VXC.Structure')
            structure.set('Compression', 'ASCII_READABLE')
            etree.SubElement(structure, "X_Voxels").text = str(x)
            etree.SubElement(structure, "Y_Voxels").text = str(y)
            etree.SubElement(structure, "Z_Voxels").text = str(z)

            for name, details in ind.genotype.to_phenotype_mapping.items():
                state = details["state"]
                flattened_state = state.reshape(z, x*y)

                data = etree.SubElement(structure, name)
                for i in range(flattened_state.shape[0]):
                    layer = etree.SubElement(data, "Layer")
                    if name == "Data":
                        str_layer = "".join([str(c) for c in flattened_state[i]])
                    # need 3d vectors for cilia
                    # else:
                    #     str_layer = "".join([str(c)+", " for c in flattened_state[i]])
                        layer.text = etree.CDATA(str_layer)

            # hack
            base_cilia_force = np.zeros((z, 3*x*y), dtype=np.float16)
            for name, details in ind.genotype.to_phenotype_mapping.items():
                state = details["state"]
                flattened_state = state.reshape(z, x*y)

                if name == "cilia_X":
                    base_cilia_force[:, ::3] = flattened_state
                if name == "cilia_Y":
                    base_cilia_force[:, 1::3] = flattened_state
                # if name == "cilia_Z":
                #     base_cilia_force[:, 2::3] = flattened_state

            data = etree.SubElement(structure, "BaseCiliaForce")
            for i in range(base_cilia_force.shape[0]):
                layer = etree.SubElement(data, "Layer")
                str_layer = "".join([str(c) + ", " for c in base_cilia_force[i]])
                layer.text = etree.CDATA(str_layer)

           # end hack


            # md5 so we don't eval the same vxd more than once
            m = hashlib.md5()
            m.update(etree.tostring(root))
            ind.md5 = m.hexdigest()

            # don't evaluate if identical phenotype has already been evaluated
            if ind.md5 in pop.already_evaluated:

                for rank, goal in pop.objective_dict.items():
                    if goal["tag"] is not None:
                        setattr(ind, goal["name"], pop.already_evaluated[ind.md5][rank])

                print "Age {0} individual already evaluated: cached fitness is {1}".format(ind.age, ind.fitness)

            else:
                # save the vxd to data folder
                with open('data'+str(seed)+'/bot_{:04d}.vxd'.format(ind.id), 'wb') as vxd:
                    vxd.write(etree.tostring(root))

    # ok let's finally evaluate all the robots in the data directory

    if record_history:  # just save history, don't assign fitness
        print "Recording the history of the run champ"
        sub.call("./Voxelyze3 -i data{0} > champ_{0}_gen{1}.hist".format(seed, pop.gen), shell=True)

    else:  # normally, we will just want to update fitness and not save the trajectory of every voxel

        print "GENERATION {}".format(pop.gen)

        print "Launching {0} voxelyze calls, out of {1} individuals".format(num_evaluated_this_gen, len(pop))


        while True:
            try:
                sub.call("./Voxelyze3 -i data{0} -o output{0}.xml".format(seed), shell=True)
                # sub.call waits for the process to return
                # after it does, we collect the results output by the simulator
                root = etree.parse("output{}.xml".format(seed)).getroot()
                break

            except IOError:
                print "Uh oh, there was an IOError! I'll re-simulate this batch again..."
                pass

            except IndexError:
                print "Uh oh, there was an IndexError! I'll re-simulate this batch again..."
                pass
           

        for ind in pop:

            if ind.phenotype.is_valid() and ind.md5 not in pop.already_evaluated:

                ind.fitness = float(root.findall("detail/bot_{:04d}/fitness_score".format(ind.id))[0].text)

                print "Assigning ind {0} fitness {1}".format(ind.id, ind.fitness)

                pop.already_evaluated[ind.md5] = [getattr(ind, details["name"])
                                                  for rank, details in
                                                  pop.objective_dict.items()]
