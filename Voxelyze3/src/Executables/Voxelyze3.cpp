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
#include <boost/process.hpp>

#include "ctool.h"

#define APP_DESCRIPTION "\
Thank you for using Voxelyze3. This program should be run on a computer that has GPUs.\n\
Typical Usage:\n\
Voxelyze -i <data_path> -o <report_path> -l -f \n\n\
<data_path> should contain a file named `base.vxa' and multiple files with extension `.vxa'.\n\
<report_path> is the report file you need to place. If you want to overwrite existing report, add -f flag.\n\
if the executable `vx3_node_worker' doesn't exist in the same path, use -w <worker> to specify the path.\n\
Allowed options\
"

int main(int argc, char** argv) {

    //setup tools for parsing arguments
    po::options_description desc(APP_DESCRIPTION);
    desc.add_options()
    ("help,h", "produce help message")
    ("locally,l", "If this machine already has GPUs, locally run tasks on this machine.")
    ("nodes,n", po::value<int>(), "The number of nodes you want to start simultaneously.")
    ("input,i", po::value<std::string>(), "Set input directory path which contains a generation of VXA files.")
    ("output,o", po::value<std::string>(), "Set output file path for report. (e.g. report_1.xml)")
    ("worker,w", po::value<std::string>(), "Specify which worker you want to use. vx3_node_worker by default.")
    ("force,f", "Overwrite output file if exists.");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);    

    //check parameters
    if (vm.count("help") || !vm.count("input") || !vm.count("output")) {
        std::cout << desc << "\n";
        return 1;
    }
    if (vm.count("nodes") && vm.count("locally")) {
        std::cout << "ERROR: -n and -s cannot use together.\n";
        std::cout << desc << "\n";
        return 1;
    }
    if (!vm.count("nodes") && !vm.count("locally")) {
        std::cout << "ERROR: must choose to run locally(-l) or envoke nodes to run(-n).\n";
        std::cout << vm.count("locally");
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
    if (!fs::is_directory(input)) {
        std::cout << "Error: input directory not found.\n\n";
        std::cout << desc << "\n";
        return 1;
    }
    if (!fs::is_regular_file(input/"base.vxa")) {
        std::cout << "No base.vxa found in input directory.\n\n";
        std::cout << desc << "\n";
        return 1;
    }
    std::string str_worker = "vx3_node_worker";
    if (vm.count("worker")) {
        str_worker = vm["worker"].as<std::string>();
    }
    fs::path worker(str_worker);
    if (!fs::is_regular_file(worker)) {
        std::cout << "Need an executable worker but nothing found.\n\n";
        std::cout << desc << "\n";
        return 1;
    }

    //Setup a workspace folder
    fs::path workspace("workspace");
    try { fs::create_directory(workspace); }
    catch(...){}

    //Do evocations: locally or distributedly
    if (vm.count("locally")) { //Produce a vxt file and pass that to vx3_node_worker
        fs::path locally(workspace/"locally");
        try {
            fs::create_directory(locally);
        } catch(...) {}
        pt::ptree tree;
        tree.put("vxa", (input/"base.vxa").string());
        tree.put("input_dir", input.string());
        for (auto & file : fs::directory_iterator( input )) {
            if (boost::algorithm::to_lower_copy(file.path().extension().string())==".vxd")
                tree.add("vxd.f", file.path().filename().string());
        }
        std::string str_time = ctool::u_format_now("%Y%m%d%H%M%S");
        std::string vxt = str_time + ".vxt";
        std::string vxr = str_time + ".vxr";
        pt::write_xml((locally/vxt).string(), tree);
        std::string command = worker.string() + " -i " + (locally/vxt).string() + " -o " + (locally/vxr).string();

        std::cout << command << "\n";
        boost::process::child process_worker(command);
        process_worker.wait();
        try {
            if (fs::is_regular_file(locally/vxr)) {
                fs::copy_file(locally/vxr, output, fs::copy_option::overwrite_if_exists);
            } else {
                printf("File not exist: %s.\n", (locally/vxr).c_str());
            }
        } catch(...) {
            printf("ERROR: Failed to copy result file: %s.\n", (locally/vxr).c_str());
        }
    } else { //Call vx3_start_daemon to check desired number of daomons are waiting on different nodes, produce multiple vxt files, monitor the results, merge into one output

    }
    return 0;
}