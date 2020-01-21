#include <iostream>
#include <boost/program_options.hpp>
namespace po = boost::program_options;
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
#include <boost/algorithm/string/case_conv.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
namespace pt = boost::property_tree;
#include <boost/foreach.hpp>

#include "VX3/VX3_SimulationManager.cuh"

/// Host ///
// Input directory
// Output file

// Determine usable nodes
// Split into batches by produce XML as a filename list
// Execute srun to start nodes and run

/// Node ///
// XML

// Determine usable GPUs
// Split into sub_batches as std::vector<std::vector<fs::path>> 
// Call SimulationManager to start

std::vector<fs::path> host(fs::path input_directory) {
    std::vector<fs::path> XMLs;
    return XMLs;
}

std::vector<std::vector<fs::path>> node(fs::path XML) {
    std::vector<std::vector<fs::path>> sub_batches;
    return sub_batches;
}

void gpu(std::vector<fs::path> sub_batches) {
    //run all fs::path in sub_batches
}


int main(int argc, char** argv) {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    if (nDevices<=0) {
        printf("Error: No GPU found.\n");
        return 1;
    } else {
        printf("%d GPUs found.\n", nDevices);
    }

    po::options_description desc(R"(
        
Usage: 
Voxelyze3 -n [num_nodes] -i <vxa_directory> -o <report_file> # this command envoke multiple GPU nodes(servers).
or
Voxelyze3 -s -i <vxa_xml> -o <report_file> # this command run directly on a GPU node(server).

Allowed options)");

    desc.add_options()
    ("help,h", "produce help message")
    ("directly,d", "Skipping starting nodes, directly run on this machine.")
    ("nodes,n", po::value<int>(), "The number of nodes you want to start simultaneously.")
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

    if (vm.count("nodes") && vm.count("directly")) {
        std::cout << "ERROR: -n and -s cannot use together.\n";
        std::cout << desc << "\n";
        return 1;
    }

    if (!vm.count("nodes") && !vm.count("directly")) {
        std::cout << "ERROR: must choose to run directly(-d) or envoke nodes to run(-n).\n";
        std::cout << vm.count("directly");
        std::cout << desc << "\n";
        return 1;
    }

    fs::path input(vm["input"].as<std::string>());
    fs::path output(vm["output"].as<std::string>());

    if (fs::is_regular_file(output) && !vm.count("force") ) {
        std::cout << "Error: output file exists.\n\n";
        std::cout << desc << "\n";
        return 1;
    }

    if (vm.count("directly")) { //run directly on node.
        printf("run directly on node.\n");
        pt::ptree tree;
        pt::read_xml(input.string(), tree);
        std::vector<std::string> filenames;
        BOOST_FOREACH(pt::ptree::value_type &v, tree.get_child("files.vxa")) {
            filenames.push_back(v.second.data());
        }
        
        for(auto f : filenames) {
            printf("%s\n", f.c_str());
        }

    } else { //envoke nodes to run
        int nodes = 1;
        if (vm.count("nodes")) {
            nodes = vm["nodes"].as<int>();
        }
        printf("Starting %d nodes.\n", nodes);
        
        if (!fs::is_directory(input)) {
            printf("Error: input directory not found.\n\n");
            std::cout << desc << "\n";
            return 1;
        }
    
    }
    





    
    



    std::cout<<"\n\n";
    return 0;
}

// std::vector<std::vector<fs::path>> splitIntoSubBatches(fs::path input_directory) { //Sub-batches are for Multiple GPUs on one node.
//     int i=0;
//     int num_of_devices;
//     cudaGetDeviceCount(&num_of_devices);
//     std::vector<std::vector<fs::path>> sub_batches;
//     sub_batches.resize(num_of_devices);
//     for (auto & file : fs::directory_iterator( input_directory )) {
//         if (boost::algorithm::to_lower_copy(file.path().extension().string()) == ".vxa") {
//             int iGPU = (i%num_of_devices);
//             sub_batches[iGPU].push_back( file.path() );
//             i++;
//         }
//     }
//     return sub_batches;
// }
