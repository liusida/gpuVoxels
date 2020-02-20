import numpy as np
import matplotlib.pyplot as plt
from lxml import etree
import shutil, os
import subprocess
from PIL import Image, ImageDraw, ImageFont

def text_phantom(text, size):
    # Availability is platform dependent
    font = 'ChunkFive-Regular.otf'

    # Create font
    pil_font = ImageFont.truetype(font, size=size // len(text),
                                  encoding="unic")
    text_width, text_height = pil_font.getsize(text)

    # create a blank canvas with extra space between lines
    canvas = Image.new('RGB', [size, size], (255, 255, 255))

    # draw the text onto the canvas
    draw = ImageDraw.Draw(canvas)
    offset = ((size - text_width) // 2,
              (size - text_height) // 2)
    white = "#000000"
    draw.text(offset, text, font=pil_font, fill=white)

    # Convert the canvas into an array with values in [0, 1]
    return (255 - np.asarray(canvas)) / 255.0

import matplotlib.pyplot as plt
alife = text_phantom('ALife', 500)
alife = alife[::-1,:,:] # img to array y is upside down
body = np.zeros([200,300,10], dtype=int)
for i in range(10):
    body[100:,:,i] = alife[200:300,100:400,0]
plt.imshow(body[:,:,0])
plt.show()
body[body==1]=2

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
exp_names = ["ALife"]
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
