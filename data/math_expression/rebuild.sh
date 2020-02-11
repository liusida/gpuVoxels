#!/bin/sh
cwd=$(pwd)
cd ../../
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
cp ../../build/Voxelyze3/Voxelyze3 .
cp ../../build/Voxelyze3/vx3_node_worker .
cp ../../build/VoxCAD/VoxCAD .
