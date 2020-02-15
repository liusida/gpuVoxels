import numpy as np
from lxml import etree
import shutil, os
import subprocess

np.random.seed(1)
voxSize = 0.01
r = np.random.random([1,300,300])
r1 = np.random.random([1,300,300])
voxels = np.zeros_like(r, dtype=int)
voxels[r>0.99] = 2
control = r1 * 2 - 1
control[voxels==0] = 0

def write_VXD(body, phaseoffset, exp_id, exp_name):
    z,y,x = body.shape
    body_flatten = body.reshape(1,-1)
    phaseoffset_flatten = phaseoffset.reshape(1,-1)
    # generate VXD
    child = etree.SubElement
    root = etree.Element("VXD")
    # Enable Attachment
    AttachDetach = child(root, "AttachDetach")
    AttachDetach.set('replace', 'VXA.Simulator.AttachDetach')
    child(AttachDetach, 'EnableCollision').text = '1'
    child(AttachDetach, 'EnableAttach').text = '1'
    child(AttachDetach, 'watchDistance').text = '1'
    child(AttachDetach, 'SafetyGuard').text = '1000'
    # Attach Condition (attach only happens when this value > 0)
    AttachCondition = child(AttachDetach, 'AttachCondition')
    formula_attach = []
    formula_attach.append("<mtCONST>1</mtCONST>")
    # attach = (x-50)^2 + (y-50)^2 - r^2
    formula_attach.append("""
    <mtSUB>
        <mtADD>
            <mtMUL>
                <mtSUB>
                    <mtVAR>x</mtVAR>
                    <mtCONST>{}</mtCONST>
                </mtSUB>
                <mtSUB>
                    <mtVAR>x</mtVAR>
                    <mtCONST>{}</mtCONST>
                </mtSUB>
            </mtMUL>
            <mtMUL>
                <mtSUB>
                    <mtVAR>y</mtVAR>
                    <mtCONST>{}</mtCONST>
                </mtSUB>
                <mtSUB>
                    <mtVAR>y</mtVAR>
                    <mtCONST>{}</mtCONST>
                </mtSUB>
            </mtMUL>
        </mtADD>
        <mtCONST>{}</mtCONST>
    </mtSUB>
    """.format(x/2*voxSize, x/2*voxSize, y/2*voxSize, y/2*voxSize, 25*voxSize*voxSize))
    formula_attach.append("""
    <mtSUB>
        <mtVAR>t</mtVAR>
        <mtCONST>3</mtCONST>
    </mtSUB>
    """)
    

    AttachCondition.append(etree.fromstring(formula_attach[exp_id]))
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
    # Gravity
    GravAcc = child(root, "GravAcc")
    GravAcc.set('replace', 'VXA.Environment.Gravity.GravAcc')
    GravAcc.text = '-1' # instead of -9.8

    # ForceField
    ForceField = child(root, "ForceField")
    ForceField.set('replace', "VXA.Simulator.ForceField")
    # x_f = y + (-100)*arctan(x-50);
    x_forcefield = child(ForceField, "x_forcefield")
    formula_x = """
    <mtADD>
        <mtMUL>
            <mtCONST>100</mtCONST>
            <mtSUB>
                <mtVAR>y</mtVAR>
                <mtCONST>{}</mtCONST>
            </mtSUB>
        </mtMUL>
        <mtMUL>
            <mtCONST>-100</mtCONST>
            <mtATAN>
                <mtSUB>
                    <mtVAR>x</mtVAR>
                    <mtCONST>{}</mtCONST>
                </mtSUB>
            </mtATAN>
        </mtMUL>
    </mtADD>
    """.format(y/2*voxSize, x/2*voxSize)
    x_forcefield.append(etree.fromstring(formula_x))
    # y_f = -x + (-100)*arctan(y-50);
    y_forcefield = child(ForceField, "y_forcefield")
    formula_y = """
    <mtADD>
        <mtMUL>
            <mtCONST>-100</mtCONST>
            <mtSUB>
                <mtVAR>x</mtVAR>
                <mtCONST>{}</mtCONST>
            </mtSUB>
        </mtMUL>
        <mtMUL>
            <mtCONST>-100</mtCONST>
            <mtATAN>
                <mtSUB>
                    <mtVAR>y</mtVAR>
                    <mtCONST>{}</mtCONST>
                </mtSUB>
            </mtATAN>
        </mtMUL>
    </mtADD>
    """.format(x/2*voxSize, y/2*voxSize)
    y_forcefield.append(etree.fromstring(formula_y))

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


# Start Experiments
exp_names = ["attach_all", "attach_area", "attach_time"]
for exp_id, exp_name in enumerate(exp_names):
    try:
        os.mkdir(exp_name)
    except:
        pass
    try:
        shutil.copyfile("./base.vxa", f"{exp_name}/base.vxa")
    except:
        print("base.vxa not found.")

    write_VXD(voxels, control, exp_id, exp_name)
