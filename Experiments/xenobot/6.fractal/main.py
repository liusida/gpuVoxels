import os, shutil, re
import numpy as np
np.random.seed(1)
import lxml.etree as etree
child = etree.SubElement
import utilities

def safe_mkdir(dir_name):
    try:
        shutil.rmtree(dir_name, ignore_errors=True)
        os.mkdir(dir_name)
    except Exception:
        pass

def toBinary(number, bit):
    return ((number & (1<<np.arange(bit))) > 0).astype(int)

# read in the 6x6x5
def read_robot():
    root = etree.parse("assets/6x6x5.data")
    morphology = []
    try:
        data = root.findall(".//Data")[0]
        phaseoffset = root.findall(".//PhaseOffset")[0]
        X_Voxels = int(root.findall(".//X_Voxels")[0].text)
        Y_Voxels = int(root.findall(".//Y_Voxels")[0].text)
        Z_Voxels = int(root.findall(".//Z_Voxels")[0].text)

    except:
        print("No valid Data.")
        exit()

    for l in data.iter("Layer"):
        layer = []
        for c in l.text:
            layer.append( 3 if c=='9' else 0 )
        morphology.append(layer)

    morphology = np.array(morphology)
    morphology = np.array(morphology).reshape(Z_Voxels, Y_Voxels, X_Voxels)

    control = []
    for l in phaseoffset.iter("Layer"):
        layer = []
        values = l.text.split(', ')[:-1]
        layer = [float(i) for i in values]
        control.append(layer)

    control = np.array(control)
    control = np.array(control).reshape(Z_Voxels, Y_Voxels, X_Voxels)
    return morphology, control

body, control = read_robot()
body1 = body.copy()
body1[body==3] = 4
z,y,x = body.shape


safe_mkdir("data")
Z = z*z; Y = y*y; X = x*x

def fractal(original_body, body, body1, control):
    c = 0
    z,y,x = original_body.shape
    Z,Y,X = body.shape
    body_2nd = np.zeros([z*Z,y*Y,x*X], dtype=int)
    body1_2nd = np.zeros([z*Z,y*Y,x*X], dtype=int)
    control_2nd = np.zeros([z*Z,y*Y,x*X], dtype=int)
    for i in range(z):
        for j in range(y):
            for k in range(x):
                if original_body[i,j,k]!=0:
                    body_2nd[i*Z:(i+1)*Z, j*Y:(j+1)*Y, k*X:(k+1)*X] = body if c%2==0 else body1
                    body1_2nd[i*Z:(i+1)*Z, j*Y:(j+1)*Y, k*X:(k+1)*X] = body if c%2==1 else body1
                    control_2nd[i*Z:(i+1)*Z, j*Y:(j+1)*Y, k*X:(k+1)*X] = control
                c+=1
            c+=1
        c+=1
    return body_2nd, body1_2nd, control_2nd
body_2nd, body1_2nd, control_2nd = fractal(body, body, body1, control)
world, _, world_control = fractal(body, body_2nd, body1_2nd, control_2nd)
Z,Y,X = world.shape

safe_mkdir(f"data")
shutil.copy("base.vxa", f"data/base.vxa")
world_flatten = world.reshape([Z,-1])
world_control_flatten = world_control.reshape([Z,-1])
root = etree.Element("VXD")

# Main Structure and PhaseOffset
structure = child(root, "Structure")
structure.set('replace', 'VXA.VXC.Structure')
structure.set('Compression', 'ASCII_READABLE')
child(structure, "X_Voxels").text = str(X)
child(structure, "Y_Voxels").text = str(Y)
child(structure, "Z_Voxels").text = str(Z)
data = child(structure, "Data")
print(world_flatten)
for i in range(world_flatten.shape[0]):
    layer = child(data, "Layer")
    str_layer = "".join([str(c) for c in world_flatten[i]])
    layer.text = etree.CDATA(str_layer)
PhaseOffset = child(structure, "PhaseOffset")
for i in range(world_control_flatten.shape[0]):
    layer = child(PhaseOffset, "Layer")
    str_layer = "".join([str(c) + ", " for c in world_control_flatten[i]])
    layer.text = etree.CDATA(str_layer)

with open(f"data/robot.vxd", 'wb') as file:
    file.write(etree.tostring(root))
