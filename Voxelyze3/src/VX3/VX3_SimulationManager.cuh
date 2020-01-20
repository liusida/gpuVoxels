#if !defined(VX3_SIMULATION_MANAGER)
#define VX3_SIMULATION_MANAGER
#include <iostream>
#include <thread>
#include <utility> 
#include <vector>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
namespace pt = boost::property_tree;

#include "VX3_VoxelyzeKernel.h"



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

class VX3_SimulationManager
{
private:
    /* data */
public:
    VX3_SimulationManager(fs::path input, fs::path output);
    ~VX3_SimulationManager();

    void start();
    void readVXA(std::vector<fs::path> files, int device_index);
    std::vector<std::vector<fs::path>> splitIntoSubBatches();
    void startKernel(int num_tasks, int device_index);
    void collectResults(int num_simulation, int device_index);
    void sortResults();
    void printResults();

    /* DATA */
    int num_of_devices; //Total number of GPUs on one single node. One DeepGreen node has 8 GPUs.
    std::vector<VX3_VoxelyzeKernel*> d_voxelyze_3s; //Multiple device memory passing to different device.
    fs::path input_directory;
    fs::path output_file;

    std::vector<VX3_SimulationResult> h_results;

};

#endif // VX3_SIMULATION_MANAGER
