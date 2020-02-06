from lxml import etree
import numpy as np

root = etree.parse("assets/robot.data")
morphology = []
control = []
try:
    data = root.findall(".//Data")[0]
    phaseoffset = root.findall(".//PhaseOffset")[0]
except:
    print("No valid Data.")
    exit()

for l in data.iter("Layer"):
    layer = []
    for c in l.text:
        layer.append( int(c) )
    morphology.append(layer)


for l in phaseoffset.iter("Layer"):
    layer = []
    numbers = l.text.split(",")
    for n in numbers:
        try:
            n = float(n.strip())
            layer.append( float(n) )
        except:
            pass
        
    control.append(layer)

morphology = np.array(morphology).reshape(5,6,6)
control = np.array(control).reshape(5,6,6)
print(control.shape , morphology.shape)
print(morphology.shape)

scaleup = 1
big_robot = morphology.repeat(scaleup, axis=0).repeat(scaleup, axis=1).repeat(scaleup, axis=2)
big_control = control.repeat(scaleup, axis=0).repeat(scaleup, axis=1).repeat(scaleup, axis=2)
print(big_robot.shape)
# exit()
z,x,y = big_robot.shape
big_robot_flatten = big_robot.reshape(z,x*y)
control_flatten = big_control.reshape(z,x*y)

# generate VXD
root = etree.Element("VXD")
# Enable Attachment
AttachDetach = etree.SubElement(root, "AttachDetach")
AttachDetach.set('replace', 'VXA.Simulator.AttachDetach')
etree.SubElement(AttachDetach, 'EnableAttach').text = '1'
etree.SubElement(AttachDetach, 'watchDistance').text = '1'
# Enable Record History
RecordHistory = etree.SubElement(root, "RecordHistory")
RecordHistory.set('replace', 'VXA.Simulator.RecordHistory')
etree.SubElement(RecordHistory, "RecordStepSize").text = '100'
# Stop Condition 2 sec
StopConditionValue = etree.SubElement(root, "StopConditionValue")
StopConditionValue.set('replace', 'VXA.Simulator.StopCondition.StopConditionValue')
StopConditionValue.text = '10'
# Main Structure and PhaseOffset
structure = etree.SubElement(root, "Structure")
structure.set('replace', 'VXA.VXC.Structure')
structure.set('Compression', 'ASCII_READABLE')
etree.SubElement(structure, "X_Voxels").text = str(x)
etree.SubElement(structure, "Y_Voxels").text = str(y)
etree.SubElement(structure, "Z_Voxels").text = str(z)
data = etree.SubElement(structure, "Data")
for i in range(big_robot_flatten.shape[0]):
    layer = etree.SubElement(data, "Layer")
    str_layer = "".join([str(c) for c in big_robot_flatten[i]])
    layer.text = etree.CDATA(str_layer)
phaseoffset = etree.SubElement(structure, "PhaseOffset")
for i in range(big_robot_flatten.shape[0]):
    layer = etree.SubElement(phaseoffset, "Layer")
    str_layer = ",".join([str(c) for c in control_flatten[i]])
    layer.text = etree.CDATA(str_layer)

with open('gen01.vxd', 'wb') as file:
    file.write(etree.tostring(root))

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = plt.axes(projection='3d')

# print(world_morphology.shape)
# xs=[]
# ys=[]
# zs=[]
# for x in range(world_morphology.shape[0]):
#     for y in range(world_morphology.shape[1]):
#         for z in range(world_morphology.shape[2]):
#             if (world_morphology[x,y,z]>0):
#                 xs.append(x)
#                 ys.append(y)
#                 zs.append(z)
# ax.scatter(xs,ys,zs)
# plt.show()