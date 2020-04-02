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

First, install `Docker CE (Community Edition) <https://docs.docker.com/install/linux/docker-ce/ubuntu/>`_.

Then, install nvidia-container:

.. code:: bash

    sudo apt-get update
    sudo apt-get install -y nvidia-container-runtime nvidia-container-toolkit

Then, start the container:

.. code:: bash

    docker run --gpus all --name=gpuVoxels --volume=/tmp:/gpuVoxels/host -it sidaliu/gpuvoxels:1.0

.. note::
    You can change the path /tmp to anywhere you want, and keep your code and data there.

.. note::
    If you are not familiar with Docker, here's some helpful commands you will use.

    .. code:: bash

        # The commands below should be run on host command line
        # See what containers are running
        docker ps
        # See what containers are running/suspending
        docker ps --all
        # You should see a suspended container called gpuVoxels
        # Let's re-enter that container
        docker start gpuVoxels
        docker attach gpuVoxels

How I made the docker image
^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you'd like to see what's inside the image **sidaliu/gpuvoxels:1.0**, here is the way I build it.

.. code:: bash

    docker run --gpus all --name=gpuVoxels --volume=/tmp:/host -it nvidia/cuda:10.1-devel-ubuntu18.04

The meaning of last command is run a docker container using nvidia image cuda, download it if not exists.
The name of container is gpuVoxels, and /tmp of the host is shared in the container as /host.

After a while...

Now, I am inside the docker container.

.. code:: bash

    apt update
    apt install -y git cmake libboost-all-dev
    cd /
    mkdir gpuVoxels
    git clone https://github.com/liusida/gpuVoxels.git gpuVoxels_src
    cd /gpuVoxels_src/Voxelyze3
    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release -DCUDA_DEBUG=OFF ..
    cmake --build .
    cp Voxelyze3 /gpuVoxels
    cp vx_node_worker /gpuVoxels
    exit
    # Back to the host

.. code:: bash

    docker commit -a "Sida Liu <sliu1@uvm.edu>" gpuVoxels sidaliu/gpuvoxels:1.0
    docker push sidaliu/gpuvoxels:1.0

On Server with GPUs but without root
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It will be too tricky to install all the dependencies without root, especially CUDA 10.x.

If there's already CUDA 10.1 or 10.2 on the server, you'll need to compile and install cmake, g++, boost from source.


Install VoxCAD
--------------

**VoxCAD** need OpenGL, Boost, etc. You will need to build VoxCAD from source. Instructions will come later.
