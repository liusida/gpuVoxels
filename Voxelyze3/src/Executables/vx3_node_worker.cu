#include <stdio.h>
#include <boost/program_options.hpp>
namespace po = boost::program_options;
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
namespace pt = boost::property_tree;
#include <boost/foreach.hpp>

#include "VX3_SimulationManager.cuh"

#define APP_DESCRIPTION "\
This application is balabalabala....\n\
Usage:\n\
xxx\n\
Allowed options\
"

std::vector<std::string> split(const std::string& s, char delimiter)
{
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter))
    {
        tokens.push_back(token);
    }
    return tokens;
}

int main(int argc, char** argv) {

    //setup tools for parsing arguments
    po::options_description desc(APP_DESCRIPTION);
    desc.add_options()
    ("help,h", "produce help message")
    ("input,i", po::value<std::string>(), "Set input directory path which contains a generation of VXA files.")
    ("output,o", po::value<std::string>(), "Set output file path for report. (e.g. report_1.xml)")
    ("force,f", "Overwrite output file if exists.");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    //check parameters
    if (vm.count("help") || !vm.count("input") || !vm.count("output")) {
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
    if (!fs::is_regular_file(input)) {
        std::cout << "Error: input file not found.\n\n";
        std::cout << desc << "\n";
        return 1;
    }


    //Read vxt file
    pt::ptree tree;
    pt::read_xml( input.string(), tree );
    fs::path base;
    fs::path input_dir;
    std::vector<fs::path> files;
    base = tree.get<fs::path>("vxa");
    input_dir = tree.get<fs::path>("input_dir");
    BOOST_FOREACH(pt::ptree::value_type &v, tree.get_child("vxd")) {
        // The data function is used to access the data stored in a node.
        files.push_back(fs::path(v.second.data()));
    }

    //count number of GPUs
    int nDevices=0;
    VcudaGetDeviceCount(&nDevices);
    if (nDevices<=0) {
        printf("Error: No GPU found.\n");
        return 1;
    } else {
        printf("%d GPU found.\n", nDevices);
    }

    //split files into sub batches (if run locally, one batch stands for all files in input directory.)
    std::vector<std::vector<fs::path>> sub_batches;
    sub_batches.resize(nDevices);

    for (int i=0;i<files.size();i++) {
        sub_batches[i%nDevices].push_back(files[i]);
    }

    VX3_SimulationManager sm(sub_batches, base, input_dir, nDevices);
    sm.start();

    pt::ptree tr_result;
    

    tr_result.put("bestfit.filename", sm.h_results[0].vxa_filename);
    tr_result.put("bestfit.distance", sm.h_results[0].distance);
    // this will be too much to write into the report.
    for (auto &res: sm.h_results) {
        std::string simulation_name = split(res.vxa_filename, '.')[0];
        tr_result.put("detail."+simulation_name+".num_voxel", res.num_voxel);
        tr_result.put("detail."+simulation_name+".CoM.x", res.x);
        tr_result.put("detail."+simulation_name+".CoM.y", res.y);
        tr_result.put("detail."+simulation_name+".CoM.z", res.z);
        tr_result.put("detail."+simulation_name+".CoM.distance_by_size", res.distance);
        // for (auto &pos: res.voxel_position) {
        //     pt::ptree t_pos;
        //     t_pos.put("x", pos.x);
        //     t_pos.put("y", pos.y);
        //     t_pos.put("z", pos.z);
        //     tr_result.add_child("detail."+simulation_name+".pos", t_pos);
        // }
    }
    pt::write_xml(output.string(), tr_result);
    return 0;
}