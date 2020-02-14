# Read a fixed morphology from assets/robot.data
#
from lxml import etree
import numpy as np
import shutil
import os
import subprocess

root = etree.parse("assets/gen230.vxc")
try:
    Data = root.findall(".//Data")[0]
    PhaseOffset = root.findall(".//PhaseOffset")[0]
    X_Voxels = int(root.findall(".//X_Voxels")[0].text)
    Y_Voxels = int(root.findall(".//Y_Voxels")[0].text)
    Z_Voxels = int(root.findall(".//Z_Voxels")[0].text)

except:
    print("No valid Data.")
    exit()

morphology = []
for l in Data.iter("Layer"):
    layer = []
    for c in l.text:
        layer.append(int(c))
    morphology.append(layer)

morphology = np.array(morphology)
morphology = np.array(morphology).reshape(Z_Voxels, Y_Voxels, X_Voxels)
morphology_flatten = morphology.reshape(Z_Voxels, -1)

control = []
for l in PhaseOffset.iter("Layer"):
    layer = []
    for c in l.text.split(","):
        layer.append(float(c))
    control.append(layer)
control = np.array(control)
control = np.array(control).reshape(Z_Voxels, Y_Voxels, X_Voxels)

# print(morphology)
z, y, x = morphology.shape

# generate random control for each voxel
#
generation_id_start = 240
num_generation = 1000
population_per_generation = 400
mutation_ratio = 0.02
best_robot = control
fitness_score = 0
base_training_time = 2
additional_training_time = 0

prefix = "generated_data_biped_v1/"
for generation_id in range(generation_id_start, generation_id_start+num_generation):
    print(f"Starting generation {generation_id}...", end="", flush=True)
    robots = []
    robots_flatten = []
    for robot_id in range(population_per_generation):
        if (best_robot is None):  # initialize
            control = np.random.random(size=morphology.shape) * 2 - 1
        else:  # mutate
            control = best_robot + \
                (np.random.random(size=morphology.shape) * 2 - 1) * mutation_ratio
        control[morphology == 0] = 0
        robots.append(control)
        control_flatten = control.reshape(Z_Voxels, -1)
        robots_flatten.append(control_flatten)

    path_gene = prefix+f"generation{generation_id:05}/"
    if path_gene[-1] != "/":
        path_gene = path_gene + "/"
    try:
        os.mkdir(prefix)
    except:
        pass
    try:
        os.mkdir(path_gene)
    except:
        pass
    for robot_id, robot in enumerate(robots_flatten):
        # robot = robots[robot_id]
        # generate VXD
        root = etree.Element("VXD")
        # Stop Condition 5 sec
        child = etree.SubElement
        StopConditionValue = child(root, "StopConditionValue")
        StopConditionValue.set(
            'replace', 'VXA.Simulator.StopCondition.StopConditionValue')
        StopConditionValue.text = str(
            base_training_time + additional_training_time)  # scaffolding!!!
        # Main Structure and PhaseOffset
        structure = child(root, "Structure")
        structure.set('replace', 'VXA.VXC.Structure')
        structure.set('Compression', 'ASCII_READABLE')
        child(structure, "X_Voxels").text = str(x)
        child(structure, "Y_Voxels").text = str(y)
        child(structure, "Z_Voxels").text = str(z)
        data = child(structure, "Data")
        for i in range(morphology_flatten.shape[0]):
            layer = child(data, "Layer")
            str_layer = "".join([str(c) for c in morphology_flatten[i]])
            layer.text = etree.CDATA(str_layer)
        phaseoffset = child(structure, "PhaseOffset")
        for i in range(robot.shape[0]):
            layer = child(phaseoffset, "Layer")
            str_layer = ",".join([str(c) for c in robot[i]])
            layer.text = etree.CDATA(str_layer)
        with open(f'{path_gene}robot_{robot_id:05}.vxd', 'wb') as file:
            file.write(etree.tostring(root))

    try:
        shutil.copyfile("./base.vxa", f"{path_gene}base.vxa")
    except:
        print("base.vxa not found.")

    running_cmd = f"./Voxelyze3 -i {path_gene} -o {path_gene}output.xml -lf"
    # print(running_cmd)
    process = subprocess.Popen(running_cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    if (error):
        print(error, flush=True)

    report = etree.parse(f"{path_gene}output.xml")
    #  data = root.findall(".//Data")[0]
    filename = report.findall(".//filename")[0].text
    fitness_score = report.findall(".//fitness_score")[0].text
    fitness_score = float(fitness_score)
    best_fit_robot_id = int(filename[len("robot_"):len("robot_")+5])
    best_robot = robots[best_fit_robot_id]
    print(f"done. best fitness score: {fitness_score}.", flush=True)

    if (fitness_score > -0.1): # allow the robot to move downward a little bit. No fall! (If the robot fall, the score will be around -0.3.)
        print("increasing trainning time to " + str(0.1+additional_training_time))
        additional_training_time += 0.1

print("Job Finished.")