import random
from lxml import etree

for k in range(100):
    root = etree.Element("PhaseOffset")
    root.set('replace', 'VXA.VXC.Structure.PhaseOffset')
    for i in range(5):
        layer = etree.SubElement(root, "Layer")
        str_random = ""
        for j in range(36):
            str_random += str(random.random()) + ","
        layer.text = etree.CDATA(str_random)

    with open('gen01-0{0:d}.vxd'.format(k), 'wb') as file:
        file.write(etree.tostring(root))
