#include "VX3/VX3_SimulationManager.cuh"
#include "VX3/VX3_VoxelyzeKernel.cuh"
// #include "VX_Sim.h"


__global__ void CUDA_Simulation(VX3_VoxelyzeKernel *d_voxelyze_3, int num_tasks) {
    // int i = blockIdx.x * blockDim.x + threadIdx.x;
    // if (i<num_tasks) {
    //     VX3_VoxelyzeKernel *d_v3 = &d_voxelyze_3[i];
    //     d_v3->syncVectors(); //Everytime we pass a class with VX3_vectors in it, we should sync hd_vector to d_vector first.
    //     printf(COLORCODE_GREEN "Simulation %d runs.\t" COLORCODE_RESET, i);
    //     for (int j=0;j<1000000;j++) { //Maximum Steps 1000000
    //         if (d_v3->StopConditionMet()) break;
    //         // if (j%1000==0) {
    //         //     printf("----> [Task %d] doTimeStep %d, Current Time (in sec) %f \t", i, j, d_v3->currentTime);
    //         //     d_v3->updateCurrentCenterOfMass();
    //         //     printf("Current Location (in meter): %f %f %f\n", d_v3->currentCenterOfMass.x, d_v3->currentCenterOfMass.y, d_v3->currentCenterOfMass.z);
    //         // }
    //         if (!d_v3->doTimeStep()) {
    //             printf(COLORCODE_BOLD_RED "\nSimulation %d Diverged.\n" COLORCODE_RESET, i);
    //             break;
    //         }
    //         // if (j% 1000==0)
    //         //     printf("Time: %f, pos[0]: %f %f %f\n", d_v3->currentTime, d_v3->d_voxels[0].pos.x, d_v3->d_voxels[0].pos.y, d_v3->d_voxels[0].pos.z);

    //     }
    //     d_v3->updateCurrentCenterOfMass();
    //     printf(COLORCODE_BLUE "Simulation %d ends.\t" COLORCODE_RESET, i);
    // }
}

void VX3_SimulationManager::operator()(fs::path batchFolder, cudaStream_t stream) {
    // if (fs::is_empty(batchFolder)) return;
    // //1. read every VXA files
    // std::vector<std::string> filenames;
    // VX3_VoxelyzeKernel * d_voxelyze_3;
    // std::vector<VX3_VoxelyzeKernel *> h_d_voxelyze_3;
    // int batch_size = 0;
    // for (auto &file : fs::directory_iterator( batchFolder) ) { batch_size++; }
    
    // VcudaMalloc((void**)&d_voxelyze_3, batch_size * sizeof(VX3_VoxelyzeKernel));
    
    // int i = 0;
    // for (auto &file : fs::directory_iterator( batchFolder ) ) {
        
    //     CVX_Environment MainEnv;
    //     CVX_Sim MainSim;
    //     CVX_Object MainObj;
    //     MainEnv.pObj = &MainObj; //connect environment to object
    //     MainSim.pEnv = &MainEnv; //connect Simulation to envirnment
    //     MainSim.LoadVXAFile(file.path().string());
    //     filenames.push_back(file.path().string());
    //     std::string err_string; //need to link this up to get info back...
    //     if (!MainSim.Import(NULL, NULL, &err_string)){
    //         std::cout<<err_string;
    //     }
        
    //     VX3_VoxelyzeKernel h_d_tmp(&MainSim.Vx, stream);
    //     h_d_tmp.DtFrac = MainSim.DtFrac;
    //     h_d_tmp.StopConditionType = MainSim.StopConditionType;
    //     h_d_tmp.StopConditionValue = MainSim.StopConditionValue;
    //     h_d_tmp.TempEnabled = MainSim.pEnv->TempEnabled;
    //     h_d_tmp.VaryTempEnabled = MainSim.pEnv->VaryTempEnabled;
    //     h_d_tmp.TempBase = MainSim.pEnv->TempBase;
    //     h_d_tmp.TempAmplitude = MainSim.pEnv->TempAmplitude;
    //     h_d_tmp.TempPeriod = MainSim.pEnv->TempPeriod;
    //     h_d_tmp.currentTemperature = h_d_tmp.TempBase + h_d_tmp.TempAmplitude;
        
    //     VcudaMemcpyAsync(d_voxelyze_3 + i, &h_d_tmp, sizeof(VX3_VoxelyzeKernel), VcudaMemcpyHostToDevice, stream);
        
    //     i++;
    // }

    // //3. start CUDA Simulation
    // int num_tasks = batch_size;
    // int threadsPerBlock = 512;
    // int numBlocks = (num_tasks + threadsPerBlock - 1) / threadsPerBlock;
    // if (numBlocks == 1)
    //     threadsPerBlock = num_tasks;
    // CUDA_Simulation<<<numBlocks,threadsPerBlock,0,stream>>>(d_voxelyze_3, num_tasks);
    // //4. wait
    // cudaStreamSynchronize(stream);

    // //5. read result
    // //sort and write normCOMdisplacement.Length()
    // //norm(current Center of mass - initial CoM / voxSize)

    // double final_z = 0.0;
    // VX3_VoxelyzeKernel* result_voxelyze_kernel = (VX3_VoxelyzeKernel *)malloc(num_tasks * sizeof(VX3_VoxelyzeKernel));
    
    // VcudaMemcpyAsync( result_voxelyze_kernel, d_voxelyze_3, num_tasks * sizeof(VX3_VoxelyzeKernel), VcudaMemcpyDeviceToHost, stream );
    
    // //TODO: how to communicate with experiments? files? or other methods?
    // printf("\n====[RESULTS for %s]====\n", batchFolder.filename().c_str());
    // std::vector< std::pair<double, int> > normAbsoluteDisplacement;
    // for (int i=0;i<num_tasks;i++) {
    //     double x = result_voxelyze_kernel[i].currentCenterOfMass.x;
    //     double y = result_voxelyze_kernel[i].currentCenterOfMass.y;
    //     double z = result_voxelyze_kernel[i].currentCenterOfMass.y;
    //     double v = result_voxelyze_kernel[i].voxSize;
    //     x = x/v; y = y/v; z = z/v;
    //     double dist = sqrt(x*x + y*y + z*z);
    //     normAbsoluteDisplacement.push_back( std::make_pair(dist,i) );
    // }
    // std::sort(normAbsoluteDisplacement.begin(), normAbsoluteDisplacement.end());
    // std::reverse(normAbsoluteDisplacement.begin(), normAbsoluteDisplacement.end());
    // pt::ptree xml_tree;
    // xml_tree.put("voxelyzeManager.batchName", batchFolder.filename());
    // for (auto p : normAbsoluteDisplacement) {
    //     pt::ptree task;
    //     task.put("normAbsoluteDisplacement", p.first);
    //     task.put("taskId", p.second);
    //     task.put("VXAFilename", filenames[p.second]);
    //     task.put("AbsoluteDistanceInMeter.x", result_voxelyze_kernel[p.second].currentCenterOfMass.x);
    //     task.put("AbsoluteDistanceInMeter.y", result_voxelyze_kernel[p.second].currentCenterOfMass.y);
    //     task.put("AbsoluteDistanceInMeter.z", result_voxelyze_kernel[p.second].currentCenterOfMass.z);
    //     task.put("VoxelSizeInMeter", result_voxelyze_kernel[p.second].voxSize);
    //     xml_tree.add_child("voxelyzeManager.Report", task);
    // }
    // pt::write_xml((batchFolder/"report.xml").string(), xml_tree, \
    //                     std::locale(), pt::xml_writer_make_settings<std::string>('\t', 1));
    // printf("Best distance of this generation is %f (x voxelSize).\n", normAbsoluteDisplacement[0].first);
    // printf("A detailed report.xml has been produced in the batch folder.\n");

    // //6. cleanup
    // for (auto p:h_d_voxelyze_3) {
    //     p->cleanup();
    // }
    // MycudaFree(d_voxelyze_3);
    // // delete result_voxelyze_kernel; //ANY PROBLEM??

    return;
}