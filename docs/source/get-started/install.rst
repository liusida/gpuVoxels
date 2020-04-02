Install
=======

Executables
-----------

Voxelyze3
^^^^^^^^^

**Voxelyze3** is an executable that split tasks into multiple groups, and call **vx_node_worker** to execute them on different GPUs.

vx_node_worker
^^^^^^^^^^^^^^

**vx_node_worker** works with **Voxelyze3** and is the one who actually carry out the calculation. (So don't delete it.)

VoxCAD
^^^^^^

**VoxCAD** can visualize **history** file recorded via **Voxelyze3**. **VoxCAD** should be run locally, and GPU is only optional for **VoxCAD**.

Install Voxelyze3
-----------------

On DeepGreen
^^^^^^^^^^^^

DeepGreen is UVM's GPU cluster. We have already depolied on DeepGreen, so it will be quite easy to use **Voxelyze3** on DeepGreen.

Follow a five-minute instruction here: `https://github.com/liusida/gpuVoxels-dg-installation <https://github.com/liusida/gpuVoxels-dg-installation>`_

On Desktop/Laptop with GPUs
^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have access to root, the best way to get started is using Docker, since Nvidia provide ready to use CUDA docker image files.

First, install `Docker<https://www.docker.com/get-started>`_ and `nvidia-docker<https://github.com/NVIDIA/nvidia-docker>`_

Then, do these:

.. code:: bash

    docker pull nvidia/cuda:10.2-devel-ubuntu18.04
    docker run --gpus all -it nvidia/cuda:10.2-devel-ubuntu18.04 /bin/bash
    # inside-docker
    apt update
    apt install git cmake libboost-all-dev
    cd /tmp
    git clone https://github.com/liusida/gpuVoxels.git
    cd gpuVoxels/Voxelyze3
    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release -DCUDA_DEBUG=OFF ..
    cmake --build .
    # ok, you now have **Voxelyze3** and **vx_node_worker** to play with.

On Server with GPUs but without root
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It will be too tricky to install all the dependencies without root, especially CUDA 10.


Install VoxCAD
--------------

**VoxCAD** need OpenGL, Boost, etc. You will need to build VoxCAD from source. Instructions will come later.

