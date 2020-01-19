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

class VX3_SimulationManager
{
private:
    /* data */
public:
    VX3_SimulationManager() = default;
    
    //Overload operator to start thread
    void operator()(fs::path batchFolder, cudaStream_t stream);


};

#endif // VX3_SIMULATION_MANAGER
