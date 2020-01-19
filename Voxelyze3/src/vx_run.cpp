#include <iostream>
#include <boost/program_options.hpp>
namespace po = boost::program_options;

int main(int argc, char** argv) {
    po::options_description desc(R"(This program divide the task into multiple batches and send them to different nodes.
        
Usage: 
vx_run -i <vxa_directory> -o <report_file>

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

    std::string input_directory = vm["input"].as<std::string>();
    std::string output_file = vm["output"].as<std::string>();
    
    std::cout << "input directory:" << input_directory <<"\n";
    std::cout << "output file:" << output_file <<"\n";

    
}
