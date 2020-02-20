import numpy as np
import matplotlib.pyplot as plt
from lxml import etree
import shutil, os
import subprocess

def generate_a_dragon(iteration=100):
    # make variables for the right and left containing 'r' and 'l' 
    r = 'r'
    l = 'l'
    # assign our first iteration a right so we can build off of it 
    old = r 
    new = old 
    # set the number of times we have been creating 
    # the next iteration as the first 
    cycle = 1
    # keep on generating the next iteration until desired iteration is reached 
    while cycle<iteration: 
        # add a right to the end of the old iteration and save it to the new 
        new = (old) + (r) 
        # flip the old iteration around(as in the first charicter becomes last) 
        old = old[::-1] 
        # cycling through each character in the flipped old iteration: 
        for char in range(0, len(old)): 
            # if the character is a right: 
            if old[char] == r: 
                # change it to a left 
                old = (old[:char])+ (l) + (old[char + 1:]) 
            # otherwise, if it's a left: 
            elif old[char] == l: 
                #change it to a right 
                old = (old[:char]) + (r) + (old[char + 1:]) 
        # add the modified old to the new iteration 
        new = (new) + (old) 
        # save the new iteration to old as well for use next cycle 
        old = new 
        # advance cycle variable to keep track of the number of times it's been done 
        cycle = cycle + 1
    return new

dragon = generate_a_dragon(10)

container_length = 100

body = np.zeros(shape=[container_length,container_length,container_length], dtype=int)

x = int(container_length/2)
y = int(container_length/2)
z = int(container_length/2)
orientation = 0 # 0 1 2 3 : up right down left
for char in dragon:
    body[x,y,z-2:z+2] = 2
    if char == 'r':
        orientation += 1
    elif char == 'l':
        orientation -= 1
    orientation = orientation % 4
    
    if orientation==0:
        y+=1
    elif orientation==1:
        x+=1
    elif orientation==2:
        y-=1
    elif orientation==3:
        x-=1
    else:
        print("ERROR: not consistent")
    if x<0 or x>=container_length or y<0 or y>=container_length:
        print("ERROR: too big")
        break

# x = int(container_length/2)
# y = int(container_length/2)
# z = int(container_length/2)
# orientation = 0 # 0 1 2 3 : up right down left
# for char in dragon:
#     body[z,y,x] = 2
#     if char == 'r':
#         orientation += 1
#     elif char == 'l':
#         orientation -= 1
#     orientation = orientation % 4
    
#     if orientation==0:
#         y+=1
#     elif orientation==1:
#         x+=1
#     elif orientation==2:
#         y-=1
#     elif orientation==3:
#         x-=1
#     else:
#         print("ERROR: not consistent")
#     if x<0 or x>=container_length or y<0 or y>=container_length:
#         print("ERROR: too big")
#         break

# plt.imshow(body[:,:,z])
# plt.show()

def write_VXD(body, phaseoffset, exp_id, exp_name):
    z,y,x = body.shape
    print(x,y,z)
    body_flatten = body.reshape(z,-1)
    phaseoffset_flatten = phaseoffset.reshape(z,-1)
    # generate VXD
    child = etree.SubElement
    root = etree.Element("VXD")
    RawPrint = child(root, "RawPrint")
    RawPrint.set('replace', 'VXA.RawPrint')
    RawPrint.text = ""
    # Enable Attachment
    AttachDetach = child(root, "AttachDetach")
    AttachDetach.set('replace', 'VXA.Simulator.AttachDetach')
    child(AttachDetach, 'EnableCollision').text = '1'
    child(AttachDetach, 'EnableAttach').text = '1'
    child(AttachDetach, 'watchDistance').text = '1'
    child(AttachDetach, 'SafetyGuard').text = '2000'
    # Stop Condition 10 sec
    StopCondition = child(root, "StopCondition")
    StopCondition.set('replace', 'VXA.Simulator.StopCondition')
    StopConditionFormula = child(StopCondition, "StopConditionFormula")
    # stop happen at (t - 10 > 0)
    stop_condition_formula = """
    <mtSUB>
        <mtVAR>t</mtVAR>
        <mtCONST>10</mtCONST>
    </mtSUB>
    """
    StopConditionFormula.append(etree.fromstring(stop_condition_formula))
    # Record History
    RecordHistory = child(root, "RecordHistory")
    RecordHistory.set('replace', 'VXA.Simulator.RecordHistory')
    child(RecordHistory, "RecordStepSize").text = '100'
    child(RecordHistory, "RecordVoxel").text = '1'
    child(RecordHistory, "RecordLink").text = '0'
    
    # Main Structure and PhaseOffset
    Structure = child(root, "Structure")
    Structure.set('replace', 'VXA.VXC.Structure')
    Structure.set('Compression', 'ASCII_READABLE')
    child(Structure, "X_Voxels").text = str(x)
    child(Structure, "Y_Voxels").text = str(y)
    child(Structure, "Z_Voxels").text = str(z)
    data = child(Structure, "Data")
    for i in range(body_flatten.shape[0]):
        layer = child(data, "Layer")
        str_layer = "".join([str(c) for c in body_flatten[i]])
        layer.text = etree.CDATA(str_layer)
    phaseoffset = child(Structure, "PhaseOffset")
    for i in range(phaseoffset_flatten.shape[0]):
        layer = child(phaseoffset, "Layer")
        str_layer = ",".join([str(c) for c in phaseoffset_flatten[i]])
        layer.text = etree.CDATA(str_layer)
    with open(f"{exp_name}/exp.vxd", 'wb') as file:
        file.write(etree.tostring(root))


world_voxels = body
world_control = np.random.random(size=body.shape)
# Start Experiments
exp_names = ["dragon"]
for exp_id, exp_name in enumerate(exp_names):
    try:
        os.mkdir(exp_name)
    except:
        pass
    try:
        shutil.copyfile("./base.vxa", f"{exp_name}/base.vxa")
    except:
        print("base.vxa not found.")

    write_VXD(world_voxels, world_control, exp_id, exp_name)
