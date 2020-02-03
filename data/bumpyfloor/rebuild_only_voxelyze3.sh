#!/bin/sh
cwd=$(pwd)
cd ../../Voxelyze3/
if [ "$1" = "-f" ]; then
    echo "Rebuild all."
    rm build/ -rf
    mkdir build
else
    echo "Just make."
fi
cd build
if [ "$1" = "-f" ]; then
    cmake -DCMAKE_BUILD_TYPE=Release ..
fi
cmake --build . -j 10
cd $cwd
cp ../../Voxelyze3/build/Voxelyze3 .
cp ../../Voxelyze3/build/vx3_node_worker .
# cp ../../build/VoxCAD/VoxCAD .
