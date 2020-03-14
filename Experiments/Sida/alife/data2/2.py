from lxml import etree
import numpy as np

world_morphology_flatten = world_morphology.reshape(z,x*y)

# generate VXD
root = etree.Element("VXD")
# Main Structure and PhaseOffset
structure = etree.SubElement(root, "Structure")
structure.set('replace', 'VXA.VXC.Structure')
structure.set('Compression', 'ASCII_READABLE')
etree.SubElement(structure, "X_Voxels").text = str(x)
etree.SubElement(structure, "Y_Voxels").text = str(y)
etree.SubElement(structure, "Z_Voxels").text = str(z)
data = etree.SubElement(structure, "Data")
for i in range(world_morphology_flatten.shape[0]):
    layer = etree.SubElement(data, "Layer")
    str_layer = "".join([str(c) for c in world_morphology_flatten[i]])
    layer.text = etree.CDATA(str_layer)

with open('data/gen01.vxd', 'wb') as file:
    file.write(etree.tostring(root))
