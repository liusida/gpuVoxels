cmake_minimum_required(VERSION 3.5.0)

add_subdirectory(Voxelyze3)

# setup enviroment
file(REMOVE_RECURSE workspace/*)
file(COPY build/Voxelyze3/Voxelyze3 DESTINATION workspace)
file(COPY build/Voxelyze3/vx3_node_worker DESTINATION workspace)
file(COPY build/Voxelyze3/vx3_node_daemon DESTINATION workspace)
