# Install Steps

Ubuntu 16.04 LTS

Install a Ubuntu 16.04 Desktop LTS to VirtualBox

User: voxcad
password: 123456

```bash
sudo apt update
sudo apt upgrade
git clone https://github.com/liusida/gpuVoxels.git
cd gpuVoxels
git checkout dev-CUDA-0.1
cd VoxCAD/
mkdir build
cd build

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

sudo apt install cmake
# install boost (C++ common library)
sudo apt install libboost-all-dev
# install qt5 (GUI Window)
sudo apt-get install qt5-default
# install GLFW3 GLUT GLM (OpenGL related)
sudo apt install libglfw3-dev
sudo apt install freeglut3-dev
sudo apt install libglm-dev
```