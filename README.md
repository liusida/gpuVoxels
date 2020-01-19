# EvoSoRo (Evolutionary Soft Robotics)

This suite includes three parts: Experiments + Voxelyze3 on DeepGreen + VoxCAD on desktop.

Here is the diagram of a typical user case:

![A Typical User Case](https://github.com/liusida/EvoSoRo/blob/master/misc/TypicalUseCase.png?raw=true)

1. Experiments (Python or other langauges) produce a generation of .VXA files (each .VXA file stands for one virtual world with one creature in it);

2. Voxelyze3 (C++ and CUDA) reads those .VXA files and start multi-threaded simulations on multiple nodes and multiple GPUs;

3. Voxelyze3 summarizes a report for this generation;

4. Experiments read the report and decide: goto 1 for another generation, or end the experiment;

5. Experiments save desired VXA files for user to download;

6. On users' laptop or desktop, VoxCAD (C++) import a VXA file;

7. With the help of included Voxelyze2 (C++), VoxCAD shows the animation on screen.
