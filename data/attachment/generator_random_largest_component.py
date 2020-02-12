import numpy as np
from scipy.ndimage.measurements import label
from lxml import etree
np.random.seed(1)
arr = np.random.random([30,30,30])>0.8
labeled_array, num_features = label(arr)
print(labeled_array)
component_sizes = []
for i in range(num_features):
    component_sizes.append(np.sum(labeled_array == i+1))
print(component_sizes)
print(np.argmax(component_sizes)+1)
largest_component = np.zeros(arr.shape, dtype=int)
largest_component[(labeled_array == (np.argmax(component_sizes)+1))] = 2
largest_component_flatten = largest_component.reshape(largest_component.shape[0],-1)


z,y,x = largest_component.shape
child = etree.SubElement
for k in range(1):
    root = etree.Element("VXD")
    # Enable Attachment
    AttachDetach = child(root, "AttachDetach")
    AttachDetach.set('replace', 'VXA.Simulator.AttachDetach')
    child(AttachDetach, 'EnableAttach').text = '1'
    child(AttachDetach, 'watchDistance').text = '1'
    child(AttachDetach, 'SafetyGuard').text = '500'
    # Enable Record History
    RecordHistory = child(root, "RecordHistory")
    RecordHistory.set('replace', 'VXA.Simulator.RecordHistory')
    child(RecordHistory, "RecordStepSize").text = '100'
    child(RecordHistory, "RecordVoxel").text = '1'
    child(RecordHistory, "RecordLink").text = '0'
    # Stop Condition 2 sec
    StopConditionValue = child(root, "StopConditionValue")
    StopConditionValue.set('replace', 'VXA.Simulator.StopCondition.StopConditionValue')
    StopConditionValue.text = '10'
    # Main Structure and PhaseOffset
    structure = child(root, "Structure")
    structure.set('replace', 'VXA.VXC.Structure')
    structure.set('Compression', 'ASCII_READABLE')
    child(structure, "X_Voxels").text = str(x)
    child(structure, "Y_Voxels").text = str(y)
    child(structure, "Z_Voxels").text = str(z)
    data = child(structure, "Data")
    for i in range(z):
        layer = child(data, "Layer")
        length = largest_component_flatten.shape[1]+2
        str_random = np.array2string(largest_component_flatten[i], separator='', max_line_width=length, threshold=length)[1:-1]
        print(str_random)
        layer.text = etree.CDATA(str_random)
    # phaseoffset = child(structure, "PhaseOffset")
    # for i in range(z):
    #     layer = child(phaseoffset, "Layer")
    #     str_random = ""
    #     for j in range(x*y):
    #         str_random += str(random.random()) + "," #random phaseoffset
    #     layer.text = etree.CDATA(str_random)

    with open('gen01-0{0:d}.vxd'.format(k), 'wb') as file:
        file.write(etree.tostring(root))
