#include <iostream>
#include <boost/program_options.hpp>
namespace po = boost::program_options;
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
#include <boost/algorithm/string/case_conv.hpp>


#include "VX3/VX3_SimulationManager.cuh"

int main(int argc, char** argv) {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    if (nDevices<=0) {
        printf("Error: No GPU found.\n");
        return 1;
    } else {
        printf("%d GPUs found.\n", nDevices);
    }

    po::options_description desc(R"(This program starts a batch of simulation on a server with multiple GPUs.
        
Usage: 
Voxelyze3 -i <vxa_directory> -o <report_file>

Allowed options)");

    desc.add_options()
    ("help,h", "produce help message")
    ("input,i", po::value<std::string>(), "Set input directory path which contains a generation of VXA files.")
    ("output,o", po::value<std::string>(), "Set output file path for report. (e.g. report_1.xml)")
    ("force,f", "Overwrite output file if exists.");
    
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);    

    if (vm.count("help") || !vm.count("input") || !vm.count("output")) {
        std::cout << desc << "\n";
        return 1;
    }

    std::string str_input_directory = vm["input"].as<std::string>();
    std::string str_output_file = vm["output"].as<std::string>();

    fs::path input_directory(str_input_directory);
    fs::path output_file(str_output_file);

    if (!fs::is_directory(input_directory)) {
        printf("Error: input directory not found.\n\n");
        std::cout << desc << "\n";
        return 1;
    }

    if (fs::is_regular_file(output_file) && !vm.count("force") ) {
        std::cout << "Error: output file exists.\n\n";
        std::cout << desc << "\n";
        return 1;
    }
    
    VX3_SimulationManager mgr(input_directory, output_file);
    mgr.start();

    std::cout<<"\n\n";
    return 0;
}
