#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <iostream>
#include <thread>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
#include <boost/asio/ip/host_name.hpp>
#include <boost/foreach.hpp> 

#define WORKSPACE "workspace"

int main() {
    //Setup a workspace folder
    fs::path workspace(WORKSPACE);
    fs::path hostname(boost::asio::ip::host_name());

    try {
        boost::filesystem::create_directory(workspace);
        boost::filesystem::create_directory(workspace/hostname);
    } catch (...) {}
    if (!fs::is_directory(workspace)) {
        std::cout << "Error: cannot create workspace, make sure you have writing permission.\n\n";
        return 1;
    }

    fs::path vxh = workspace/hostname/"heartbeats.vxh";

    FILE *fp;
    time_t now;
    while(1) {
        //Monitor
        // BOOST_FOREACH(auto const &p, fs::directory_iterator(workspace/hostname))   
        // { 
        //     printf("%s", p.path().filename().c_str());
        //     if(fs::is_regular_file(p))
        //     {
        //         // do something with p
        //     } 
        // }

        //Heart beat
        time(&now);
        fp = fopen(vxh.c_str(), "w");
        fprintf(fp, "%ld", now);
        fclose(fp);
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
}


/*
[sliu1@dg-user1 build]$ sinfo -N
NODELIST      NODES PARTITION STATE 
dg-gpunode01      1    dggpu* idle  
dg-gpunode02      1    dg-jup idle  
dg-gpunode03      1    dggpu* mix   
dg-gpunode04      1    dggpu* idle  
dg-gpunode05      1    dggpu* idle  
dg-gpunode06      1    dggpu* idle  
dg-gpunode07      1    dggpu* idle  
dg-gpunode08      1    dggpu* idle  
dg-gpunode09      1    dggpu* idle  
dg-gpunode10      1    dggpu* idle 


[sliu1@dg-user1 build]$ srun --nodelist=dg-gpunode01 ./vx3_node_daemon 
srun: Required node not available (down, drained or reserved)

*/