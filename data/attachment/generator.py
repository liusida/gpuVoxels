import random
from lxml import etree

x = 5
y = 5
z = 200
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
    child(RecordHistory, "RecordStepSize").text = '400'
    child(RecordHistory, "RecordVoxel").text = '1'
    child(RecordHistory, "RecordLink").text = '1'
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
        str_random = ""
        for j in range(x*y):
            if random.random()>0.8: # random morphology
                str_random += '2'
            else:
                str_random += '0'
        layer.text = etree.CDATA(str_random)
    phaseoffset = child(structure, "PhaseOffset")
    for i in range(z):
        layer = child(phaseoffset, "Layer")
        str_random = ""
        for j in range(x*y):
            str_random += str(random.random()) + "," #random phaseoffset
        layer.text = etree.CDATA(str_random)

    with open('gen01-0{0:d}.vxd'.format(k), 'wb') as file:
        file.write(etree.tostring(root))
