#include <iostream>

struct VX3_SimulationResult {
    double x;
    double y;
    double z;
    double voxSize;
    double distance; //a unitless distance
    std::string vxa_filename;

    void computeDisplacement() {
        distance = sqrt(x*x + y*y + z*z)/voxSize;
    }
    static bool compareDistance(VX3_SimulationResult i1, VX3_SimulationResult i2) // for sorting results
    { 
        return (i1.distance < i2.distance); 
    } 
};