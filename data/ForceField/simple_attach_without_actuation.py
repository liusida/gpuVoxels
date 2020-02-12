from lxml import etree
import numpy as np

world = np.zeros(shape=[1,10,10], dtype=int)
world[0,0,9] = 9
world[0,1,3] = 3
world[0,0,5] = 3
world[0,1,5] = 3
world_flatten = world.reshape([1, 100])

# generate VXD
root = etree.Element("VXD")
child = etree.SubElement  # a shortcut
# Enable Attachment
AttachDetach = child(root, "AttachDetach")
AttachDetach.set('replace', 'VXA.Simulator.AttachDetach')
child(AttachDetach, 'EnableAttach').text = '1'
child(AttachDetach, 'watchDistance').text = '1'
# Enable Record History
RecordHistory = child(root, "RecordHistory")
RecordHistory.set('replace', 'VXA.Simulator.RecordHistory')
child(RecordHistory, "RecordStepSize").text = '50'
# Stop Condition 2 sec
StopConditionValue = child(root, "StopConditionValue")
StopConditionValue.set(
    'replace', 'VXA.Simulator.StopCondition.StopConditionValue')
StopConditionValue.text = '5'
# ForceField
ForceField = child(root, "ForceField")
ForceField.set('replace', "VXA.Simulator.ForceField")
x_forcefield = child(ForceField, "x_forcefield")
sub = child(x_forcefield, "mtSUB")
child(sub, "mtCONST").text = "0"
mul = child(sub, "mtMUL")
child(mul, "mtCONST").text = "100"
child(mul, "mtVAR").text = "x"
y_forcefield = child(ForceField, "y_forcefield")
sub = child(y_forcefield, "mtSUB")
child(sub, "mtCONST").text = "0"
mul = child(sub, "mtMUL")
child(mul, "mtCONST").text = "100"
child(mul, "mtVAR").text = "y"
# Main Structure and PhaseOffset
structure = child(root, "Structure")
structure.set('replace', 'VXA.VXC.Structure')
structure.set('Compression', 'ASCII_READABLE')
child(structure, "X_Voxels").text = str(world.shape[1])
child(structure, "Y_Voxels").text = str(world.shape[2])
child(structure, "Z_Voxels").text = str(world.shape[0])
data = child(structure, "Data")
print(world_flatten)
for i in range(world_flatten.shape[0]):
    layer = child(data, "Layer")
    str_layer = "".join([str(c) for c in world_flatten[i]])
    layer.text = etree.CDATA(str_layer)

with open('gen01.vxd', 'wb') as file:
    file.write(etree.tostring(root))
