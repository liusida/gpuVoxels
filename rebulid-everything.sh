if [ "$#" -eq  "1" ] # arg num is 1
then
    rm build/ -rf
    mkdir build
fi
rm workspace/ -rf
# mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j 10
cd ..
mkdir workspace
cp build/Voxelyze3/Voxelyze3 workspace/
cp build/Voxelyze3/vx3_* workspace/

