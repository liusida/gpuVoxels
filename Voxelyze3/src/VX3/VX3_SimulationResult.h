#include <iostream>
#include "Vec3D.h"

struct VX3_SimulationResult {
    double x;
    double y;
    double z;
    double voxSize;
    int num_voxel;
    // double distance; //a unitless distance
    // double distance_xy;
    double fitness_score; //fitness score defined in VXD file.
    std::string vxa_filename;
    std::vector<int> voxel_mats;
    std::vector<Vec3D<>> voxel_init_pos;
    std::vector<Vec3D<>> voxel_position;
    Vec3D<> initialCenterOfMass;
    Vec3D<> currentCenterOfMass;
    int numClosePairs = 0;

    static bool compareFitnessScore(VX3_SimulationResult i1, VX3_SimulationResult i2) // for sorting results
    {
        // Diverged.
        if (isnan(i2.fitness_score)) return true;
        if (isnan(i1.fitness_score)) return false;
        // Not Diverged.
        return (i1.fitness_score > i2.fitness_score);
    } 
};