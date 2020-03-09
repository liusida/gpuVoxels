from lxml import etree
import numpy as np
root= etree.parse("data100/bot_1306.vxd")
data = root.findall(".//Data")[0]
m100 = []

for l in data.iter("Layer"):
    layer = []
    for c in l.text:
        layer.append( int(c) )
    m100.append(layer)

m100 = np.array(m100)

m100 = m100.reshape([100,100,100])[:,:,::-1]

root = etree.parse("bot_26937.vxd")

data = root.findall(".//Data")[0]
m40 = []

for l in data.iter("Layer"):
    layer = []
    for c in l.text:
        layer.append( int(c) )
    m40.append(layer)

m40 = np.array(m40)

m40 = m40.reshape([40,40,40])[:,:,::-1]


root = etree.parse("5x6x6.data")

data = root.findall(".//Data")[0]
m6 = []

for l in data.iter("Layer"):
    layer = []
    for c in l.text:
        layer.append( 2 if int(c)==9 else 0 )
    m6.append(layer)

m6 = np.array(m6)

m6 = m6.reshape([5,6,6])

x = 200;  y = z = 100

world_morphology = np.zeros([z,y,x],dtype=int)

world_morphology[:,:,50:150] = m100
world_morphology[:40,:40,:40] = m40
world_morphology[:5,20:26,160:166] = m6
print(world_morphology.shape)
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
