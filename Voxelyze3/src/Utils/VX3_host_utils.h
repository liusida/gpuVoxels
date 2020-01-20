#if !defined(VX3_UTILS_H)
#define VX3_UTILS_H

#include <chrono>
#include <iomanip>
#include <ctime>
#include <sstream>
#include <string>

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
#include <boost/algorithm/string.hpp>

inline std::string u_format_now(std::string format) {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    std::stringstream folderName;
    folderName << std::put_time(std::localtime(&in_time_t), format.c_str());
    return folderName.str();
}

inline bool u_with_ext(fs::path file, std::string ext) {

    std::string ext_file = file.filename().extension().string();
    boost::to_upper(ext);
    boost::to_upper(ext_file);

    return ext==ext_file;
}

#endif // VX3_UTILS_H
