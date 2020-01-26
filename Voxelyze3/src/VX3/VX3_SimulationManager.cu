#include <boost/algorithm/string/case_conv.hpp>
#include "VX3_SimulationManager.cuh"
#include "ctool.h"

#include "VX3_VoxelyzeKernel.cuh"
#include "VX_Sim.h" //readVXA


__global__ void CUDA_Simulation(VX3_VoxelyzeKernel *d_voxelyze_3, int num_simulation, int device_index) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<num_simulation) {
        VX3_VoxelyzeKernel *d_v3 = &d_voxelyze_3[i];
        d_v3->syncVectors(); //Everytime we pass a class with VX3_vectors in it, we should sync hd_vector to d_vector first.
        printf(COLORCODE_GREEN "%d) Simulation %d runs: %s. with stop condition: %f. \n" COLORCODE_RESET, device_index, i, d_v3->vxa_filename, d_v3->StopConditionValue);
        printf("%d) Simulation %d: links %d, voxels %d.\n", device_index, i, d_v3->num_d_links, d_v3->num_d_voxels);
        for (int j=0;j<1000000;j++) { //Maximum Steps 1000000
            if (d_v3->StopConditionMet()) break;
            if (!d_v3->doTimeStep()) {
                printf(COLORCODE_BOLD_RED "\n%d) Simulation %d Diverged: %s.\n" COLORCODE_RESET, device_index, i, d_v3->vxa_filename);
                break;
            }
        }
        d_v3->updateCurrentCenterOfMass();
        printf(COLORCODE_BLUE "%d) Simulation %d ends: %s Time: %f, pos[0]: %f %f %f\n" COLORCODE_RESET, device_index, i, d_v3->vxa_filename, d_v3->currentTime, d_v3->currentCenterOfMass.x, d_v3->currentCenterOfMass.y, d_v3->currentCenterOfMass.z);
    }
}

VX3_SimulationManager::VX3_SimulationManager(std::vector<std::vector<fs::path>> in_sub_batches, fs::path in_base, fs::path in_input_dir, int in_num_of_devices) : 
sub_batches(in_sub_batches), base(in_base),
num_of_devices(in_num_of_devices), input_dir(in_input_dir) {
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
    for (int device_index=0;device_index<num_of_devices;device_index++) { //multi GPUs
        auto files = sub_batches[device_index];
        cudaSetDevice(device_index);
        printf("=== set device to %d for %ld simulations ===\n", device_index, files.size());
        // readVXA(base)
        readVXD(base, files, device_index);
        startKernel(files.size(), device_index);
    }
    cudaDeviceSynchronize();
    for (int device_index=0;device_index<num_of_devices;device_index++) { //multi GPUs
        auto files = sub_batches[device_index];
        collectResults(files.size(), device_index);
    }
    sortResults();
}

void VX3_SimulationManager::readVXD(fs::path base, std::vector<fs::path> files, int device_index) {
    pt::ptree pt_baseVXA;
    pt::read_xml(base.string(), pt_baseVXA);

    int num_simulation = files.size();
    
    VcudaMalloc((void**)&d_voxelyze_3s[device_index], num_simulation * sizeof(VX3_VoxelyzeKernel));
    
    int i = 0;
    for (auto &file : files ) {
        // Read VXD file, clone base VXA, replace parts specified in VXD, send to MainSim.ReadVXA to process.
        printf("reading %s\n", (input_dir/file).c_str());
        pt::ptree pt_VXD;
        pt::read_xml( (input_dir/file).string(), pt_VXD );
        pt::ptree pt_merged = pt_baseVXA;
        ctool::ptree_merge(pt_VXD, pt_merged);
        std::ostringstream stream_merged;
        std::string str_merged;
        pt::write_xml(stream_merged, pt_merged);
        str_merged = stream_merged.str();
        CXML_Rip XML;
        XML.fromXMLText(&str_merged);
        CVX_Environment MainEnv;
        CVX_Sim MainSim;
        CVX_Object MainObj;
        MainEnv.pObj = &MainObj; //connect environment to object
        MainSim.pEnv = &MainEnv; //connect Simulation to envirnment
        MainSim.ReadVXA(&XML);

        std::string err_string; //need to link this up to get info back...
        if (!MainSim.Import(NULL, NULL, &err_string)){
            std::cout<<err_string;
        }
        VX3_VoxelyzeKernel h_d_tmp(&MainSim);
        strcpy(h_d_tmp.vxa_filename, file.filename().c_str());
        VcudaMemcpy(d_voxelyze_3s[device_index] + i, &h_d_tmp, sizeof(VX3_VoxelyzeKernel), cudaMemcpyHostToDevice);
        i++;
    }
}
    // TODO: Read more and more VXD ourselves. But I'll do this later.
        // //read VXD to h_d_tmp;
        // printf("reading %s\n", (input_dir/file).c_str());
        // pt::ptree tree;
        // pt::read_xml( (input_dir/file).string(), tree );

        // h_d_tmp.voxSize            = tree.get<double>  ("VXA.VXC.Lattice.Lattice_Dim", h_d_tmp.voxSize); //lattice size
        // h_d_tmp.DtFrac             = tree.get<double>  ("VXA.Simulator.Integration.DtFrac", h_d_tmp.DtFrac);
        // h_d_tmp.StopConditionType  = (StopCondition) tree.get<int>     ("VXA.Simulator.StopCondition.StopConditionType", (int) h_d_tmp.StopConditionType);
        // h_d_tmp.StopConditionValue = tree.get<double>     ("VXA.Simulator.StopCondition.StopConditionValue", h_d_tmp.StopConditionValue);

        // h_d_tmp.TempEnabled        = tree.get<int>     ("VXA.Environment.Thermal.TempEnabled", h_d_tmp.TempEnabled); //overall flag for temperature calculations
        // h_d_tmp.VaryTempEnabled    = tree.get<int>     ("VXA.Environment.Thermal.VaryTempEnabled", h_d_tmp.VaryTempEnabled); //is periodic variation of temperature on?
        // h_d_tmp.TempBase           = tree.get<int>     ("VXA.Environment.Thermal.TempBase", h_d_tmp.TempBase);
        // h_d_tmp.TempAmplitude      = tree.get<double>  ("VXA.Environment.Thermal.TempAmplitude", h_d_tmp.TempAmplitude);
        // h_d_tmp.TempPeriod         = tree.get<double>  ("VXA.Environment.Thermal.TempPeriod", h_d_tmp.TempPeriod); //degress celcius
        
        // //TODO: FIRST PRIORITY: Read material, read voxels, read phaseoffset, make links.
        
    //
    // pt::ptree &tree_material = tree.get_child("VXC.Palette");
    // VcudaMalloc( (void **)&h_d_base.d_linkMats, tree_material.size() * sizeof(TI_MaterialLink));
    // int i=0;
    // for (const auto &v: tree_material) {
    //     double youngsModulus = v.get<double> ("Material.Mechanical.Elastic_Mod");
    //     double density = v.get<double> ("Material.Mechanical.Density");
    //     CVX_MaterialLink tmp_material(youngsModulus, density);
    //     tmp_material.myname = v.get<std::string> ("Material.Name");
    //     tmp_material.alphaCTE = v.get<double> ("Material.Mechanical.CTE");
    //     TI_MaterialLink tmp_linkMat(tmp_material);
    //     cudaMemcpy(h_d_base.d_linkMats + i, &tmp_linkMat, sizeof(TI_MaterialLink), cudaMemcpyHostToDevice);
    //     i++;
    // }

    // int X_Voxels = tree.get<int>("VXA.VXC.Structure.X_Voxels");
    // int Y_Voxels = tree.get<int>("VXA.VXC.Structure.Y_Voxels");
    // int Z_Voxels = tree.get<int>("VXA.VXC.Structure.Z_Voxels");
    // pt::ptree &tree_data_layers = tree.get_child("VXA.VXC.Structure.Data.Layer");
    // std::vector<std::string> data_layers;
    // if (tree_data_layers.size()!=Z_Voxels) {
    //     printf("Voxel layer data not present or does not match expected size.");
    // }
    // for (const auto &v : tree_data_layers) {
    //     std::string DataIn;
    //     std::string RawData = v.second.data();
    //     DataIn.resize(RawData.size());
    //     for (int i=0; i<(int)RawData.size(); i++){
    //         DataIn[i] = RawData[i]-48; //if compressed using this scheme
    //     }
    //     if (DataIn.length() != X_Voxels*Y_Voxels){
    //         printf("Voxel layer data not present or does not match expected size.");
    //     }
    //     for(int k=0; k<X_Voxels*Y_Voxels; k++){
    //         //the object's internal representation at this stage is as a long array, starting at (x0,xy,z0), proceeding to (xn,y0,z0), next to (xn,yn,z0), and on to (xn,yn,zn)
    //         h_d_base.d_linkMats + i
    //         SetData(X_Voxels*Y_Voxels*i + k, DataIn[k]); //pDataIn[k];
    //     }
    // }


void VX3_SimulationManager::startKernel(int num_simulation, int device_index) {
    int threadsPerBlock = 512;
    int numBlocks = (num_simulation + threadsPerBlock - 1) / threadsPerBlock;
    if (numBlocks == 1)
        threadsPerBlock = num_simulation;
    // printf("Starting kernel on device %d. passing d_voxelyze_3s[device_index] %p.\n", device_index, d_voxelyze_3s[device_index]);
    VX3_VoxelyzeKernel* result_voxelyze_kernel = (VX3_VoxelyzeKernel *)malloc(num_simulation * sizeof(VX3_VoxelyzeKernel));
    VcudaMemcpy( result_voxelyze_kernel, d_voxelyze_3s[device_index], num_simulation * sizeof(VX3_VoxelyzeKernel), cudaMemcpyDeviceToHost );
    CUDA_Simulation<<<numBlocks,threadsPerBlock>>>(d_voxelyze_3s[device_index], num_simulation, device_index);
    CUDA_CHECK_AFTER_CALL();
}

void VX3_SimulationManager::collectResults(int num_simulation, int device_index)  {
    //insert results to h_results
    VX3_VoxelyzeKernel* result_voxelyze_kernel = (VX3_VoxelyzeKernel *)malloc(num_simulation * sizeof(VX3_VoxelyzeKernel));
    VcudaMemcpy( result_voxelyze_kernel, d_voxelyze_3s[device_index], num_simulation * sizeof(VX3_VoxelyzeKernel), cudaMemcpyDeviceToHost );
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
