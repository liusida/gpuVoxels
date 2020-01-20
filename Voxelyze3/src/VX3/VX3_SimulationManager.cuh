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
    void writeResults(int num_tasks);

    /* DATA */
    int num_of_devices;
    std::vector<VX3_VoxelyzeKernel*> d_voxelyze_3s;
    fs::path input_directory;
    fs::path output_file;

};

#endif // VX3_SIMULATION_MANAGER
