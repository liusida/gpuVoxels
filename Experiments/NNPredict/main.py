#%%
# Read a fixed morphology from assets/robot.data
#
from lxml import etree
import numpy as np

root = etree.parse("assets/robot.data")
morphology = []
try:
    data = root.findall(".//Data")[0]
except:
    print("No valid Data.")
    exit()

for l in data.iter("Layer"):
    layer = []
    for c in l.text:
        layer.append( int(c) )
    morphology.append(layer)

morphology = np.array(morphology)
morphology = np.array(morphology).reshape(5,6,6)
morphology_flatten = morphology.reshape(5,-1)
print(morphology)
z,y,x = morphology.shape

#%%
# generate random control for each voxel
#
for robot_id in range(100):
    control = np.random.random(size=morphology.shape)  * 2 - 1
    control[morphology==0]=0
    control_flatten = control.reshape(5,-1)
    # generate VXD
    root = etree.Element("VXD")
    # Stop Condition 2 sec
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
    for i in range(control_flatten.shape[0]):
        layer = etree.SubElement(phaseoffset, "Layer")
        str_layer = ",".join([str(c) for c in control_flatten[i]])
        layer.text = etree.CDATA(str_layer)

    with open(f'generation01/robot_{robot_id:03}.vxd', 'wb') as file:
        file.write(etree.tostring(root))

import shutil 
try:
    shutil.copyfile("./base.vxa", "./generation01/base.vxa")
except:
    print("base.vxa not found.")
