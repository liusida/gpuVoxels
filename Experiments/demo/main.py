from lxml import etree
import numpy as np
np.random.seed(2)
world = np.random.random(size=[100,5,5])
world[world<0.8]=0
world[world>=0.8]=2
world = world.astype(int)
world_flatten = world.reshape([100, 25])

# generate VXD
root = etree.Element("VXD")
child = etree.SubElement  # a shortcut
# Enable Attachment
AttachDetach = child(root, "AttachDetach")
AttachDetach.set('replace', 'VXA.Simulator.AttachDetach')
child(AttachDetach, 'EnableAttach').text = '1'
child(AttachDetach, 'OnlyEnableCollision').text = '1' # not implemented yet
child(AttachDetach, 'watchDistance').text = '1'
child(AttachDetach, 'SafetyGuard').text = '500'

# Stop Condition 2 sec
StopConditionValue = child(root, "StopConditionValue")
StopConditionValue.set(
    'replace', 'VXA.Simulator.StopCondition.StopConditionValue')
StopConditionValue.text = '10'

# Enable Record History
RecordHistory = child(root, "RecordHistory")
RecordHistory.set('replace', 'VXA.Simulator.RecordHistory')
child(RecordHistory, "RecordStepSize").text = '100'
child(RecordHistory, "RecordVoxel").text = '1'
child(RecordHistory, "RecordLink").text = '0'

# ForceField
ForceField = child(root, "ForceField")
ForceField.set('replace', "VXA.Simulator.ForceField")
x_forcefield = child(ForceField, "x_forcefield")
a = child(x_forcefield, "mtADD")
m = child(a, "mtMUL")
child(m, "mtCONST").text = "-190"
child(m, "mtVAR").text = "x"
m = child(a, "mtMUL")
child(m, "mtCONST").text = "90"
child(m, "mtVAR").text = "y"

y_forcefield = child(ForceField, "y_forcefield")
a = child(y_forcefield, "mtADD")
m = child(a, "mtMUL")
child(m, "mtCONST").text = "-90"
child(m, "mtVAR").text = "x"
m = child(a, "mtMUL")
child(m, "mtCONST").text = "-190"
child(m, "mtVAR").text = "y"

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
