#include <boost/algorithm/string/case_conv.hpp>
#include "VX3/VX3_SimulationManager.cuh"
#include "VX3_VoxelyzeKernel.h"
#include "VX_Sim.h"


__global__ void CUDA_Simulation(VX3_VoxelyzeKernel *d_voxelyze_3, int num_simulation, int device_index) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<num_simulation) {
        VX3_VoxelyzeKernel *d_v3 = &d_voxelyze_3[i];
        d_v3->syncVectors(); //Everytime we pass a class with VX3_vectors in it, we should sync hd_vector to d_vector first.
        printf(COLORCODE_GREEN "%d) Simulation %d runs: %s. \n" COLORCODE_RESET, device_index, i, d_v3->vxa_filename);
        for (int j=0;j<1000000;j++) { //Maximum Steps 1000000
            if (d_v3->StopConditionMet()) break;
            if (!d_v3->doTimeStep()) {
                printf(COLORCODE_BOLD_RED "\n%d) Simulation %d Diverged: %s.\n" COLORCODE_RESET, device_index, i, d_v3->vxa_filename);
                break;
            }
        }
        d_v3->updateCurrentCenterOfMass();
        printf(COLORCODE_BLUE "%d) Simulation %d ends: %s Time: %f, pos[0]: %f %f %f\n" COLORCODE_RESET, device_index, i, d_v3->vxa_filename, d_v3->currentTime, d_v3->d_voxels[0].pos.x, d_v3->d_voxels[0].pos.y, d_v3->d_voxels[0].pos.z);
    }
}

VX3_SimulationManager::VX3_SimulationManager(fs::path input, fs::path output) : 
input_directory(input), output_file(output) {
    cudaGetDeviceCount(&num_of_devices);
    
    d_voxelyze_3s.resize(num_of_devices);
    for (int i=0;i<num_of_devices;i++) {
        d_voxelyze_3s[i] = NULL;
    }
}
VX3_SimulationManager::~VX3_SimulationManager() {
    for (auto d:d_voxelyze_3s) {
        if (d) VcudaFree(d);
    }
}

void VX3_SimulationManager::start() {
    std::vector<std::vector<fs::path>> sub_batches = splitIntoSubBatches();
    
    for (int device_index=0;device_index<num_of_devices;device_index++) { //multi GPUs
        auto files = sub_batches[device_index];
        cudaSetDevice(device_index);
        printf("=== set device to %d for %ld simulations ===\n", device_index, files.size());
        readVXA(files, device_index);
        startKernel(files.size(), device_index);
    }
    cudaDeviceSynchronize();
    for (int device_index=0;device_index<num_of_devices;device_index++) { //multi GPUs
        auto files = sub_batches[device_index];
        collectResults(files.size(), device_index);
    }
    sortResults();
}

void VX3_SimulationManager::readVXA(std::vector<fs::path> files, int device_index) {
    std::vector<std::string> filenames;
    int num_simulation = files.size();
    
    VcudaMalloc((void**)&d_voxelyze_3s[device_index], num_simulation * sizeof(VX3_VoxelyzeKernel));
    
    int i = 0;
    for (auto &file : files ) {
        
        CVX_Environment MainEnv;
        CVX_Sim MainSim;
        CVX_Object MainObj;
        MainEnv.pObj = &MainObj; //connect environment to object
        MainSim.pEnv = &MainEnv; //connect Simulation to envirnment
        MainSim.LoadVXAFile(file.string());
        filenames.push_back(file.string());
        std::string err_string; //need to link this up to get info back...
        if (!MainSim.Import(NULL, NULL, &err_string)){
            std::cout<<err_string;
        }
        
        VX3_VoxelyzeKernel h_d_tmp(&MainSim.Vx, 0);
        //not all the data needed is in MainSim.Vx, so here are several other assignments:
        strcpy(h_d_tmp.vxa_filename, file.filename().c_str());
        h_d_tmp.DtFrac = MainSim.DtFrac;
        h_d_tmp.StopConditionType = MainSim.StopConditionType;
        h_d_tmp.StopConditionValue = MainSim.StopConditionValue;
        h_d_tmp.TempEnabled = MainSim.pEnv->TempEnabled;
        h_d_tmp.VaryTempEnabled = MainSim.pEnv->VaryTempEnabled;
        h_d_tmp.TempBase = MainSim.pEnv->TempBase;
        h_d_tmp.TempAmplitude = MainSim.pEnv->TempAmplitude;
        h_d_tmp.TempPeriod = MainSim.pEnv->TempPeriod;
        h_d_tmp.currentTemperature = h_d_tmp.TempBase + h_d_tmp.TempAmplitude;
        
        // printf("copy %s to device %d.\n", h_d_tmp.vxa_filename, device_index);
        VcudaMemcpyAsync(d_voxelyze_3s[device_index] + i, &h_d_tmp, sizeof(VX3_VoxelyzeKernel), VcudaMemcpyHostToDevice, 0);
        
        i++;
    }
}

std::vector<std::vector<fs::path>> VX3_SimulationManager::splitIntoSubBatches() { //Sub-batches are for Multiple GPUs on one node.
    int i=0;
    std::vector<std::vector<fs::path>> sub_batches;
    sub_batches.resize(num_of_devices);
    for (auto & file : fs::directory_iterator( input_directory )) {
        if (boost::algorithm::to_lower_copy(file.path().extension().string()) == ".vxa") {
            int iGPU = (i%num_of_devices);
            sub_batches[iGPU].push_back( file.path() );
            i++;
        }
    }
    return sub_batches;
}

void VX3_SimulationManager::startKernel(int num_simulation, int device_index) {
    int threadsPerBlock = 512;
    int numBlocks = (num_simulation + threadsPerBlock - 1) / threadsPerBlock;
    if (numBlocks == 1)
        threadsPerBlock = num_simulation;
    // printf("Starting kernel on device %d. passing d_voxelyze_3s[device_index] %p.\n", device_index, d_voxelyze_3s[device_index]);
    CUDA_Simulation<<<numBlocks,threadsPerBlock>>>(d_voxelyze_3s[device_index], num_simulation, device_index);
    CUDA_CHECK_AFTER_CALL();
}

void VX3_SimulationManager::collectResults(int num_simulation, int device_index)  {
    //insert results to h_results
    VX3_VoxelyzeKernel* result_voxelyze_kernel = (VX3_VoxelyzeKernel *)malloc(num_simulation * sizeof(VX3_VoxelyzeKernel));
    VcudaMemcpy( result_voxelyze_kernel, d_voxelyze_3s[device_index], num_simulation * sizeof(VX3_VoxelyzeKernel), VcudaMemcpyDeviceToHost );
    for (int i=0;i<num_simulation;i++) {
        VX3_SimulationResult tmp;
        tmp.x = result_voxelyze_kernel[i].currentCenterOfMass.x;
        tmp.y = result_voxelyze_kernel[i].currentCenterOfMass.y;
        tmp.z = result_voxelyze_kernel[i].currentCenterOfMass.z;
        tmp.voxSize = result_voxelyze_kernel[i].voxSize;
        tmp.vxa_filename = result_voxelyze_kernel[i].vxa_filename;
        tmp.computeDisplacement();
        h_results.push_back(tmp);
    }
}

void VX3_SimulationManager::printResults() {
    for (auto &r : h_results) {
        printf("%s, dis: %f.\n", r.vxa_filename.c_str(), r.distance);
    }
}

void VX3_SimulationManager::sortResults() {
    sort(h_results.begin(), h_results.end(), VX3_SimulationResult::compareDistance); 
}
