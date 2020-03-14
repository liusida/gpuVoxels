# All operations related to subfolders and specific files should be put in this file
# So the structure is managed here only
#
from .helper import *
import lxml.etree as etree
child = etree.SubElement

Voxelyze3 = "./simulator/Voxelyze3"
vx3_node_worker = "./simulator/vx3_node_worker"

def foldername_generation(experiment_name, generation):
    return f"data/experiment_{experiment_name}/generation_{generation:04}"

def clear_workspace():
    import shutil, os
    if os.path.exists("workspace"):
        shutil.rmtree("workspace")

def prepare_directories(experiment_name, generation):
    mkdir_if_not_exist("data")
    mkdir_if_not_exist(f"data/experiment_{experiment_name}")
    base_dir = foldername_generation(experiment_name, generation)
    mkdir_if_not_exist(base_dir)
    sub_dir = ["start_population", "report", "mutation_model", "bestfit"]
    for d in sub_dir:
        mkdir_if_not_exist(f"{base_dir}/{d}")
    return base_dir

def copy_vxa(experiment_name, generation):
    import shutil
    foldername = foldername_generation(experiment_name, generation)
    shutil.copy("assets/base.vxa", f"{foldername}/start_population/base.vxa")

def write_vxd(experiment_name, generation, robot_id, body, phaseoffset=None):
    vxd_filename = f"data/experiment_{experiment_name}/generation_{generation:04}/start_population/robot_{robot_id:04}.vxd"
    Z,Y,X = body.shape
    xRoot = etree.Element("VXD")
    # Main Structure and PhaseOffset
    xStructure = child(xRoot, "Structure")
    xStructure.set('replace', 'VXA.VXC.Structure')
    xStructure.set('Compression', 'ASCII_READABLE')
    child(xStructure, "X_Voxels").text = str(X)
    child(xStructure, "Y_Voxels").text = str(Y)
    child(xStructure, "Z_Voxels").text = str(Z)
    body_flatten = body.reshape([Z,-1])
    xData = child(xStructure, "Data")
    for i in range(body_flatten.shape[0]):
        layer = child(xData, "Layer")
        str_layer = "".join([str(c) for c in body_flatten[i]])
        layer.text = etree.CDATA(str_layer)
    if phaseoffset is not None:
        phaseoffset_flatten = phaseoffset.reshape([Z,-1])
        xPhaseOffset = child(xStructure, "PhaseOffset")
        for i in range(phaseoffset_flatten.shape[0]):
            layer = child(xPhaseOffset, "Layer")
            str_layer = ",".join([f"{c:.03f}" for c in phaseoffset_flatten[i]])
            layer.text = etree.CDATA(str_layer)
    with open(vxd_filename, 'wb') as file:
        file.write(etree.tostring(xRoot))

def read_report(experiment_name, generation):
    import re
    report_filename = f"{foldername_generation(experiment_name, generation)}/report/output.xml"
    report = etree.parse(report_filename)
    detail = report.xpath("/report/detail")[0]
    sorted_result = {"id": [], "fitness": []}
    # read all detail. robot_id and fitness.
    for robot in detail:
        robot_id = int(re.search(r'\d+', robot.tag).group())
        fitness = float(robot.xpath("fitness_score")[0].text)
        sorted_result["id"].append(robot_id)
        sorted_result["fitness"].append(fitness)
    return sorted_result

def copy_and_add_recordset(src, dst):
    best_fit = etree.parse(src)
    vxd = best_fit.xpath("/VXD")[0]
    RecordStepSize = child(vxd, "RecordStepSize")
    RecordStepSize.set("replace", "VXA.Simulator.RecordHistory.RecordStepSize")
    RecordStepSize.text = "100"
    with open(dst, "wb") as file:
        file.write(etree.tostring(best_fit))

def record_bestfit_history(experiment_name, generation):
    import shutil
    foldername = foldername_generation(experiment_name, generation)
    report_filename = f"{foldername}/report/output.xml"
    history_foldername = f"{foldername}/bestfit/"
    report = etree.parse(report_filename)
    best_fit_filename = report.xpath("/report/bestfit/filename")[0].text
    #vxd
    copy_and_add_recordset(f"{foldername}/start_population/{best_fit_filename}", f"{history_foldername}/{best_fit_filename}")
    #vxa
    shutil.copy("assets/base.vxa", f"{history_foldername}/base.vxa")
    #run (for convenience, we use linux pipeline here. if you use windows, please modify accrodingly.)
    commandline = f"{Voxelyze3} -i {history_foldername} -w {vx3_node_worker} > {history_foldername}/bestfit.history"
    run_shell_command(commandline)

def start_simulator(experiment_name, generation):
    # pipe output to log files
    foldername = foldername_generation(experiment_name, generation)
    commandline = f"{Voxelyze3} -i {foldername}/start_population/ -w {vx3_node_worker} -o {foldername}/report/output.xml -lf >> logs/{experiment_name}.log"
    run_shell_command(commandline)

def load_last_generation(experiment_name):
    import os, re
    import numpy as np
    max_generation_number = 0
    max_genration_foldername = ""
    if os._exists(f"data/experiment_{experiment_name}/"):
        folders = os.listdir(f"data/experiment_{experiment_name}/")
        for folder in folders:
            g = re.findall("[0-9]+", folder)
            if len(g)>=1:
                g = int(g[0])
                if g>max_generation_number:
                    max_generation_number = g
                    max_genration_foldername = folder
    if max_generation_number==0:
        # previous generation not found
        return None,0
    population = {"body": [], "phaseoffset": []}
    max_genration_foldername = f"data/experiment_{experiment_name}/{max_genration_foldername}/start_population/"
    for filename in os.listdir(max_genration_foldername):
        if filename[-4:]==".vxd":
            xRoot = etree.parse(f"{max_genration_foldername}/{filename}")
            x = int(xRoot.xpath("/VXD/Structure/X_Voxels")[0].text)
            y = int(xRoot.xpath("/VXD/Structure/Y_Voxels")[0].text)
            z = int(xRoot.xpath("/VXD/Structure/Z_Voxels")[0].text)
            #Body
            Layers = xRoot.xpath("/VXD/Structure/Data/Layer")
            lines = []
            for layer in Layers:
                line = []
                for ch in layer.text:
                    line.append(int(ch))
                lines.append(line)
            lines = np.array(lines)
            population["body"].append(lines.reshape([z,y,x]))
            #PhaseOffset
            Layers = xRoot.xpath("/VXD/Structure/PhaseOffset/Layer")
            lines = []
            for layer in Layers:
                line = []
                for ch in layer.text.split(","):
                    line.append(float(ch))
                lines.append(line)
            lines = np.array(lines)
            population["phaseoffset"].append(lines.reshape([z,y,x]))
    return population, max_generation_number

def empty_population_like(population):
    ret = {}
    for key in population:
        ret[key] = []
    return ret

def add_population(src, dst):
    anykey = list(src.keys())[0]
    for i in range(len(src[anykey])):
        for key in src.keys():
            dst[key].append(src[key][i])

if __name__ == "__main__":
    def test_write_vxd():
        import shutil
        import numpy as np
        body = np.zeros([3,3,3], dtype=int)
        body[0,1,1:] = 1
        body[0,0,0] = 1
        body[2,2,2] = 1
        mkdir_if_not_exist("tmp")
        write_vxd("tmp/1.vxd", body, body)
        shutil.rmtree("tmp")
    # test_write_vxd()