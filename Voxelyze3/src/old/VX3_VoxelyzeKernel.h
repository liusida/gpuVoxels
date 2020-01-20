#if !defined(VX3_VOXELYZE_KERNEL_H)
#define VX3_VOXELYZE_KERNEL_H
#include "VX3.h"
#include "Voxelyze.h"
#include "TI_Link.h"
#include "TI_Voxel.h"
#include "TI_MaterialLink.h"
#include "VX_Enums.h"

class VX3_VoxelyzeKernel
{   
public:
    /* Host methods */
    VX3_VoxelyzeKernel(CVoxelyze* In, cudaStream_t In_stream);
    void cleanup();
    TI_MaterialLink * getMaterialLink(CVX_MaterialLink* vx_mats);

    /* Cuda methods */
    __device__ bool doTimeStep(float dt = -1.0f);
    __device__ double recommendedTimeStep();
    __device__ void updateCurrentCenterOfMass();
    __device__ bool StopConditionMet();
    __device__ void updateTemperature();
    __device__ void syncVectors();

    /* data */
	double voxSize; //lattice size
	double currentTime = 0.0f; //current time of the simulation in seconds
    double OptimalDt = 0.0f;
    double DtFrac;
    StopCondition StopConditionType;
    double StopConditionValue;
    unsigned long CurStepCount = 0.0f;
    
    //Temperature:
	bool TempEnabled; //overall flag for temperature calculations
	bool VaryTempEnabled; //is periodic variation of temperature on?
	double TempBase, TempAmplitude, TempPeriod; //degress celcius
	double currentTemperature; //updated based on time... (for phase 0... individual materials now have their own current temp

    TI_Vec3D<double> currentCenterOfMass;

    std::vector<CVX_Link *> h_links;
    TI_Link * d_links;
    int num_d_links;

    std::vector<CVX_Voxel *> h_voxels;
    TI_Voxel * d_voxels;
    int num_d_voxels;

    std::vector<CVX_MaterialLink *> h_linkMats;
    TI_MaterialLink * d_linkMats;
    int num_d_linkMats;

    bool* d_collisionsStale;
    // TI_vector<TI_Collision *>* d_collisions;
    // TI_vector<TI_Collision *> h_collisions;

    /* CUDA Stream */
    cudaStream_t stream;
};


#endif // VX3_VOXELYZE_KERNEL_H
