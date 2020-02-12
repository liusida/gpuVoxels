#!/bin/sh
cwd=$(pwd)
BuildVoxelyze3=true
BuildVoxCAD=true
RebuildAll=false

while getopts “:f3c” opt; do
  case $opt in #-f means rebuild from scratch
    f) RebuildAll=true ;;
  esac
  case $opt in #-c means only build VoxCAD
    c) BuildVoxelyze3=false ;;
  esac
  case $opt in #-3 means only build Voxelyze3
    3) BuildVoxCAD=false ;;
  esac
done


if $BuildVoxelyze3; then
    # For Voxelyze3 (Server side)
    cd ../../Voxelyze3/
    if $RebuildAll; then
        echo "Rebuilding Voxelyze3."
        rm build/ -rf
        mkdir build
    else
        echo "Making Voxelyze3."
    fi
    cd build
    if $RebuildAll; then
        cmake -DCMAKE_BUILD_TYPE=Release -DCUDA_DEBUG=OFF ..
    fi
    cmake --build . -j 10
    cd $cwd
    cp ../../Voxelyze3/build//Voxelyze3 .
    cp ../../Voxelyze3/build/vx3_node_worker .
fi

if $BuildVoxCAD; then
    # For VoxCAD (Client side)
    cd ../../VoxCAD/
    if $RebuildAll; then
        echo "Rebuilding VoxCAD."
        rm build/ -rf
        mkdir build
    else
        echo "Making VoxCAD."
    fi
    cd build
    if $RebuildAll; then
        cmake -DCMAKE_BUILD_TYPE=Release -DCUDA_DEBUG=OFF ..
    fi
    cmake --build . -j 10
    cd $cwd
    cp ../../VoxCAD/build/VoxCAD .
fi