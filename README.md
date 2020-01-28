# gpuVoxels

(We need a well written introduction with pics. But basically this is a tool for us run experiments on Deep Green.)

## Typical User Case

This suite includes three parts: Experiments + Voxelyze3 on DeepGreen + VoxCAD on desktop.

Here is the diagram of a typical user case:

![A Typical User Case](https://github.com/liusida/gpuVoxels/blob/master/doc/misc/TypicalUseCase.png?raw=true)

1. Experiments (Python or other langauges) produce a generation of .VXA files (each .VXA file stands for one virtual world with one creature in it);

2. Voxelyze3 (C++ and CUDA) reads those .VXA files and start multi-threaded simulations on multiple nodes and multiple GPUs;

3. Voxelyze3 summarizes a report for this generation;

4. Experiments read the report and decide: goto 1 for another generation, or end the experiment;

5. Experiments save desired VXA files for user to download;

6. On users' laptop or desktop, VoxCAD (C++) import a VXA file;

7. With the help of included Voxelyze2 (C++), VoxCAD shows the animation on screen.

## VXA File

VXA file is an important interface in this project. VXA is in XML format.

[Here is a specification of .VXA file.](https://github.com/liusida/gpuVoxels/blob/master/doc/VXA_File_Format.md)

[Here is an example of .VXA file.](https://github.com/liusida/gpuVoxels/blob/master/doc/misc/example.vxa)

## Installation

### Install on DeepGreen

If you are on DeepGreen, copy my `~/apps` folder and `~/share/bin` folder, and add those three lines to your `~/.bashrc`:

```bash
# add for Voxelyze3
export LD_LIBRARY_PATH=LD_LIBRARY_PATH:/usr/local/cuda/lib64/:~/apps/lib64
export CMAKE_CXX_COMPILER=/users/s/l/sliu1/apps/bin/g++
export CMAKE_CC_COMPILER=/users/s/l/sliu1/apps/bin/gcc
```
and done.

`./Voxelyze3 -i data -o output.xml -lf`

should do the work.

### Install Elsewhere

There are several prerequisites:

```bash
CUDA 10.1
cmake-3.16.2
boost_1_66_0 #with all components (Other components may be used later.)
gcc-8
g++-8
```

After you installed those, clone the repo and use cmake to compile:

```bash
mkdir Voxelyze3/build
cd Voxelyze3/build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j 10
```

Now you have your `Voxelyze3` and `vx3_node_worker`, copy that to anywhere you want and run `Voxelyze3` as an entry point.
