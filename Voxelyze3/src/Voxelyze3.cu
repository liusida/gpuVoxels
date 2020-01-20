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
    ("output,o", po::value<std::string>(), "Set output file path for report. (e.g. report_1.xml)");
    
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

    // std::cout << "input directory:" << input_directory.string() <<"\n";
    // std::cout << "output file:" << output_file.string() <<"\n";
    int i=0;
    std::vector<std::vector<fs::path>> sub_batches;
    sub_batches.resize(nDevices);
    for (auto & file : fs::directory_iterator( input_directory )) {
        if (boost::algorithm::to_lower_copy(file.path().extension().string()) == ".vxa") {
            int iGPU = (i%nDevices);
            sub_batches[iGPU].push_back( file.path() );
        }
    }

    for (auto &files : sub_batches) {
        printf("=====%ld====\n", files.size());
        for (auto &file : files) {
            
            std::cout << file.string() <<"\n";

        }
    }
}
