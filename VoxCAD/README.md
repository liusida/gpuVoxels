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
# change default g++ to g++-8
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 10
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 10
# install cmake 3.5+
sudo apt install cmake
# install boost
sudo apt install libboost-all-dev

# install 
wget https://github.com/Kitware/CMake/releases/download/v3.16.2/cmake-3.16.2.tar.gz
tar -xf cmake-3.16.2.tar.gz
cd cmake-3.16.2
./bootstrap --parallel=10 
```