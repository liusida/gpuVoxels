import random
from lxml import etree

x = 40
y = 40
z = 40
for k in range(1):
    root = etree.Element("VXD")
    RecordHistory = etree.SubElement(root, "RecordHistory")
    RecordHistory.set('replace', 'VXA.Simulator.RecordHistory')
    etree.SubElement(RecordHistory, "RecordStepSize").text = '100'
    # VXA.Simulator.StopCondition.StopConditionValue = 0.5
    StopConditionValue = etree.SubElement(root, "StopConditionValue")
    StopConditionValue.set('replace', 'VXA.Simulator.StopCondition.StopConditionValue')
    StopConditionValue.text = '0.5'

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
            str_random += '9'
        layer.text = etree.CDATA(str_random)
    phaseoffset = etree.SubElement(structure, "PhaseOffset")
    for i in range(z):
        layer = etree.SubElement(phaseoffset, "Layer")
        str_random = ""
        for j in range(x*y):
            str_random += str(random.random()) + ","
        layer.text = etree.CDATA(str_random)

    with open('gen01-0{0:d}.vxd'.format(k), 'wb') as file:
        file.write(etree.tostring(root))
