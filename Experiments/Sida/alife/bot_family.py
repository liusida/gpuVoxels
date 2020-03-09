from lxml import etree
import numpy as np

def read_robot(size=10, orientation=True):
    root= etree.parse(f"data{size}/bot.vxd")
    data = root.findall(".//Data")[0]
    m = []
    for l in data.iter("Layer"):
        layer = []
        for c in l.text:
            layer.append( int(c) )
        m.append(layer)
    m = np.array(m)
    m = m.reshape([size,size,size])
    if orientation:
        m = np.swapaxes(m,0,1)
    return m

m10 = read_robot(10)
m20 = read_robot(20)
m30 = read_robot(30,False)
m40 = read_robot(40,False)
m50 = read_robot(50,False)
m100 = read_robot(100,False)

x = 300;  y = z = 160

world_morphology = np.zeros([z,y,x],dtype=int)

def insert_robot(m, p = [0,0,0]):
    global world_morphology
    world_morphology[p[0]:p[0]+m.shape[0], p[1]:p[1]+m.shape[1], p[2]:p[2]+m.shape[2]] = m

insert_robot(m100, [0,0,160])
insert_robot(m50, [0,100,0])
insert_robot(m40, [0,0,0])
insert_robot(m30, [0,100,80])
insert_robot(m20, [0,20,90])
insert_robot(m10, [0,40,60])

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
