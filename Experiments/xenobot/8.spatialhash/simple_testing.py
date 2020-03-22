import os, shutil, re
import numpy as np
np.random.seed(1)
import lxml.etree as etree
child = etree.SubElement

def safe_mkdir(dir_name):
    try:
        shutil.rmtree(dir_name, ignore_errors=True)
        os.mkdir(dir_name)
    except Exception:
        pass

def toBinary(number, bit):
    return ((number & (1<<np.arange(bit))) > 0).astype(int)

scale = 1
world = np.zeros([5,scale,scale], dtype=int)
for i in range(scale):
    for j in range(scale):
        if (i+j)%2==0:
            world[4,i,j] = 1
            world[0,i,j] = 1

Z,Y,X = world.shape

safe_mkdir(f"data")
shutil.copy("base.vxa", f"data/base.vxa")
world_flatten = world.reshape([Z,-1])
root = etree.Element("VXD")

# Main Structure and PhaseOffset
structure = child(root, "Structure")
structure.set('replace', 'VXA.VXC.Structure')
structure.set('Compression', 'ASCII_READABLE')
child(structure, "X_Voxels").text = str(X)
child(structure, "Y_Voxels").text = str(Y)
child(structure, "Z_Voxels").text = str(Z)
data = child(structure, "Data")
print(world_flatten)
for i in range(world_flatten.shape[0]):
    layer = child(data, "Layer")
    str_layer = "".join([str(c) for c in world_flatten[i]])
    layer.text = etree.CDATA(str_layer)

with open(f"data/robot.vxd", 'wb') as file:
    file.write(etree.tostring(root))
