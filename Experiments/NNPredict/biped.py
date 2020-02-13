#%%
# Read a fixed morphology from assets/robot.data
#
from lxml import etree
import numpy as np
import shutil, os
import subprocess

root = etree.parse("assets/Biped.vxc")
morphology = []
try:
    data = root.findall(".//Data")[0]
    X_Voxels = int(root.findall(".//X_Voxels")[0].text)
    Y_Voxels = int(root.findall(".//Y_Voxels")[0].text)
    Z_Voxels = int(root.findall(".//Z_Voxels")[0].text)

except:
    print("No valid Data.")
    exit()

for l in data.iter("Layer"):
    layer = []
    for c in l.text:
        layer.append( int(c) )
    morphology.append(layer)

morphology = np.array(morphology)
morphology = np.array(morphology).reshape(Z_Voxels, Y_Voxels, X_Voxels)
morphology_flatten = morphology.reshape(Z_Voxels,-1)
# print(morphology)
z,y,x = morphology.shape

#%%
# generate random control for each voxel
#
num_generation = 1000
population_per_generation = 100
mutation_ratio = 0.02
best_robot = None

prefix = "generated_data_biped_v1/"
for generation_id in range(num_generation):
    print(f"Starting generation {generation_id}..." , end="", flush=True)
    robots = []
    robots_flatten = []
    for robot_id in range(population_per_generation):
        if (best_robot is None):#initialize
            control = np.random.random(size=morphology.shape)  * 2 - 1
        else:#mutate
            control = best_robot + (np.random.random(size=morphology.shape)  * 2 - 1) * mutation_ratio
        control[morphology==0]=0
        robots.append(control)
        control_flatten = control.reshape(Z_Voxels,-1)
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
        StopConditionValue = etree.SubElement(root, "StopConditionValue")
        StopConditionValue.set('replace', 'VXA.Simulator.StopCondition.StopConditionValue')
        StopConditionValue.text = '5'
        # Main Structure and PhaseOffset
        structure = etree.SubElement(root, "Structure")
        structure.set('replace', 'VXA.VXC.Structure')
        structure.set('Compression', 'ASCII_READABLE')
        etree.SubElement(structure, "X_Voxels").text = str(x)
        etree.SubElement(structure, "Y_Voxels").text = str(y)
        etree.SubElement(structure, "Z_Voxels").text = str(z)
        data = etree.SubElement(structure, "Data")
        for i in range(morphology_flatten.shape[0]):
            layer = etree.SubElement(data, "Layer")
            str_layer = "".join([str(c) for c in morphology_flatten[i]])
            layer.text = etree.CDATA(str_layer)
        phaseoffset = etree.SubElement(structure, "PhaseOffset")
        for i in range(robot.shape[0]):
            layer = etree.SubElement(phaseoffset, "Layer")
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
    #%%
    #  data = root.findall(".//Data")[0]
    filename = report.findall(".//filename")[0].text
    distance = report.findall(".//distance")[0].text
    best_fit_robot_id = int(filename[len("robot_"):len("robot_")+5])
    best_robot = robots[best_fit_robot_id]
    print(f"done. best fit distance: {distance}.", flush=True)
    # %%
