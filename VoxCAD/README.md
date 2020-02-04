# Installation

This is an example of installing ONLY the VoxCAD on a clean Ubuntu 16.04 LTS.

(Voxelyze3 will need GPU and CUDA, so usually on our laptop, we only need to install VoxCAD.)

Once you have the VoxCAD, you can it anywhere and use it to visualize `.history` files in `../data` using command line `./VoxCAD something.history`.

```bash
sudo apt update
sudo apt upgrade
# clone the repo
git clone https://github.com/liusida/gpuVoxels.git
cd gpuVoxels
git checkout dev-CUDA-0.1

# install gcc g++ 8.0
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
sudo apt update
sudo apt install -y gcc-8 g++-8 
# change default g++ to g++-8 (Compilers)
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 10
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 10
# install cmake 3.16 (Building tool)
wget https://github.com/Kitware/CMake/releases/download/v3.16.3/cmake-3.16.3-Linux-x86_64.tar.gz
tar -xf cmake-3.16.3-Linux-x86_64.tar.gz
cd cmake-3.16.3-Linux-x86_64
sudo cp bin /usr/ -r
sudo cp man /usr/share/ -r
sudo cp share /usr/ -r
sudo mkdir /usr/lib/cmake
# install glm
sudo apt install libglm-dev
cd VoxCAD/
sudo mkdir /usr/lib/cmake/glm
cp cmake/* /usr/lib/cmake/glm/ #glm need to be placed manually. wired!

# install boost (C++ common library)
sudo apt install libboost-all-dev
# install qt5 (Window GUI)
sudo apt-get install qt5-default
# install GLFW3 GLUT GLM (OpenGL related)
sudo apt install libglfw3-dev
sudo apt install freeglut3-dev

# Finally, ready to build!
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j 10
```

![VoxCAD](https://github.com/liusida/gpuVoxels/blob/dev-CUDA-0.1/doc/misc/VoxCAD.png?raw=true)