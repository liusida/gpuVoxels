x=4; y=4; z=3
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

safe_mkdir("data")
safe_mkdir("bestfit")

# very first generation, initialization
robots = []
for robot_id in range(10):
    r = np.random.random([z,y,x])
    r1 = np.random.random([z,y,x])
    r1[1:z-1, 1:y-1, 1:x-1] = 1
    # print(r1[0].astype(int))
    body = np.zeros_like(r, dtype=int)
    body[r<0.5] = 5
    body[r>=0.5] = 4
    body[r1<0.4] = 0
    body[np.logical_not(utilities.make_one_shape_only(body))]=0
    robots.append(body)

for generation_id in range(5):
    safe_mkdir(f"data/gen{generation_id:04}")
    shutil.copy("base.vxa", f"data/gen{generation_id:04}/base.vxa")

    for robot_id, body in enumerate(robots):
        body_flatten = body.reshape([z,-1])

        root = etree.Element("VXD")

        # Main Structure and PhaseOffset
        structure = child(root, "Structure")
        structure.set('replace', 'VXA.VXC.Structure')
        structure.set('Compression', 'ASCII_READABLE')
        child(structure, "X_Voxels").text = str(x)
        child(structure, "Y_Voxels").text = str(y)
        child(structure, "Z_Voxels").text = str(z)
        data = child(structure, "Data")
        print(body_flatten)
        for i in range(body_flatten.shape[0]):
            layer = child(data, "Layer")
            str_layer = "".join([str(c) for c in body_flatten[i]])
            layer.text = etree.CDATA(str_layer)

        with open(f"data/gen{generation_id:04}/robot{robot_id:04}.vxd", 'wb') as file:
            file.write(etree.tostring(root))
    os.system(f"./Voxelyze3 -i data/gen{generation_id:04} -o data/gen{generation_id:04}/output.xml -lf")
    output = etree.parse(f"data/gen{generation_id:04}/output.xml")
    
    # Best Fit
    best_fit_filename = output.xpath("/report/bestfit/filename")[0].text
    
    best_fit = etree.parse(f"data/gen{generation_id:04}/{best_fit_filename}")
    vxd = best_fit.xpath("/VXD")[0]
    RecordStepSize = child(vxd, "RecordStepSize")
    RecordStepSize.set("replace", "VXA.Simulator.RecordHistory.RecordStepSize")
    RecordStepSize.text = "100"
    safe_mkdir(f"bestfit/gen{generation_id:04}")
    shutil.copy("base.vxa", f"bestfit/gen{generation_id:04}/base.vxa")

    with open(f"bestfit/gen{generation_id:04}/{best_fit_filename}", "wb") as file:
        file.write(etree.tostring(best_fit))
    
    os.system(f"./Voxelyze3 -i bestfit/gen{generation_id:04}/ > bestfit/gen{generation_id:04}/a.history")

    # Next Generation
    detail = output.xpath("/report/detail")[0]
    # choose first half
    detail = detail[:int(len(detail)/2)]
    robots_next_generation = []
    for robot in detail:
        robot_id = int(re.search(r'\d+', robot.tag).group())
        robots_next_generation.append(robots[robot_id])
    # replicate second half with mutation
    mutation_rate = 0.1
    assert(mutation_rate<0.3)
    
    robots = []
    for robot in robots_next_generation:
        r = np.random.random([z,y,x])
        r0 = np.zeros_like(r, dtype=int) # all 0
        r4 = np.ones_like(r, dtype=int) * 4 # all 4
        r5 = np.ones_like(r, dtype=int) * 5 # all 5
        filter1 = r<mutation_rate
        filter2 = np.logical_and(r>mutation_rate, r<mutation_rate*2)
        filter3 = np.logical_and(r>mutation_rate*2, r<mutation_rate*3)
        robot[filter1] = r4[filter1]
        robot[filter2] = r5[filter2]
        robot[filter3] = r0[filter3]
        robot[np.logical_not(utilities.make_one_shape_only(robot))]=0
        robots.append(robot)
    robots = [*robots, *robots_next_generation]
