import random
from lxml import etree

x = 10
y = 10
z = 10
for k in range(1):
    root = etree.Element("VXD")
    # Enable Attachment
    AttachDetach = etree.SubElement(root, "AttachDetach")
    AttachDetach.set('replace', 'VXA.Simulator.AttachDetach')
    etree.SubElement(AttachDetach, 'EnableAttach').text = '1'
    etree.SubElement(AttachDetach, 'watchDistance').text = '1'
    # Enable Record History
    RecordHistory = etree.SubElement(root, "RecordHistory")
    RecordHistory.set('replace', 'VXA.Simulator.RecordHistory')
    etree.SubElement(RecordHistory, "RecordStepSize").text = '20'
    # Stop Condition 2 sec
    StopConditionValue = etree.SubElement(root, "StopConditionValue")
    StopConditionValue.set('replace', 'VXA.Simulator.StopCondition.StopConditionValue')
    StopConditionValue.text = '2'
    # Main Structure and PhaseOffset
    structure = etree.SubElement(root, "Structure")
    structure.set('replace', 'VXA.VXC.Structure')
    structure.set('Compression', 'ASCII_READABLE')
    etree.SubElement(structure, "X_Voxels").text = str(x)
    etree.SubElement(structure, "Y_Voxels").text = str(y)
    etree.SubElement(structure, "Z_Voxels").text = str(z)
    data = etree.SubElement(structure, "Data")
    for i in range(z):
        layer = etree.SubElement(data, "Layer")
        str_random = ""
        for j in range(x*y):
            if random.random()>0.8: # random morphology
                str_random += '9'
            else:
                str_random += '0'
        layer.text = etree.CDATA(str_random)
    phaseoffset = etree.SubElement(structure, "PhaseOffset")
    for i in range(z):
        layer = etree.SubElement(phaseoffset, "Layer")
        str_random = ""
        for j in range(x*y):
            str_random += str(random.random()) + "," #random phaseoffset
        layer.text = etree.CDATA(str_random)

    with open('gen01-0{0:d}.vxd'.format(k), 'wb') as file:
        file.write(etree.tostring(root))
