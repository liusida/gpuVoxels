#include <iostream>
#include "Vec3D.h"

struct VX3_SimulationResult {
    double x;
    double y;
    double z;
    double voxSize;
    int num_voxel;
    double distance; //a unitless distance
    double distance_xy;
    std::string vxa_filename;
    std::vector<Vec3D<double>> voxel_position;
    Vec3D<double> initialCenterOfMass;
    Vec3D<double> currentCenterOfMass;

    void computeDisplacement() {
        // distance = sqrt(x*x + y*y + z*z)/voxSize;
        Vec3D<double> init_xy = initialCenterOfMass;
        init_xy.z = 0;
        Vec3D<double> current_xy = currentCenterOfMass;
        current_xy.z = 0;
        distance = currentCenterOfMass.Dist(initialCenterOfMass) / voxSize;
        distance_xy = current_xy.Dist(init_xy) /voxSize;
    }
    static bool compareDistance(VX3_SimulationResult i1, VX3_SimulationResult i2) // for sorting results
    {
        // Diverged.
        if (isnan(i2.distance)) return true;
        if (isnan(i1.distance)) return false;
        // Not Diverged.
        return (i1.distance > i2.distance);
    } 
};