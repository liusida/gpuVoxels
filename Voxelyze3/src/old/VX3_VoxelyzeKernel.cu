#include "VX3_VoxelyzeKernel.h"
#include "VX3_MemoryCleaner.h"

/* Sub GPU Threads */
__global__ void gpu_update_force(TI_Link* links, int num);
__global__ void gpu_update_voxel(TI_Voxel* voxels, int num, double dt);
__global__ void gpu_update_temperature(TI_Voxel* voxels, int num, double currentTemperature);

/* Host methods */

VX3_VoxelyzeKernel::VX3_VoxelyzeKernel(CVoxelyze* In, cudaStream_t In_stream)
{
    stream = In_stream;

    voxSize = In->voxSize;
    
    num_d_linkMats = In->linkMats.size();
    VcudaMalloc( (void **)&d_linkMats, num_d_linkMats * sizeof(TI_MaterialLink));
    {
        int i = 0;
        for (auto mat:In->linkMats) {
            TI_MaterialLink tmp_linkMat( mat, stream );
            VcudaMemcpyAsync( d_linkMats+i, &tmp_linkMat, sizeof(TI_MaterialLink), VcudaMemcpyHostToDevice, stream );
            h_linkMats.push_back( mat );
            i++;
        }
    }

    num_d_voxels = In->voxelsList.size();
    VcudaMalloc( (void **)&d_voxels, num_d_voxels * sizeof(TI_Voxel));
    for (int i=0;i<num_d_voxels;i++) {
        h_voxels.push_back( In->voxelsList[i] );
    }

    num_d_links = In->linksList.size();
    VcudaMalloc( (void **)&d_links, num_d_links * sizeof(TI_Link));
    for (int i=0;i<num_d_links;i++) {
        TI_Link tmp_link( In->linksList[i], this );
        VcudaMemcpyAsync( d_links+i, &tmp_link, sizeof(TI_Link), VcudaMemcpyHostToDevice, stream );
        h_links.push_back( In->linksList[i] );
    }

    for (int i=0;i<num_d_voxels;i++) {
        //set values for GPU memory space
        TI_Voxel tmp_voxel(In->voxelsList[i], this);
        VcudaMemcpyAsync(d_voxels+i, &tmp_voxel, sizeof(TI_Voxel), VcudaMemcpyHostToDevice, stream);
    }

    // VcudaMalloc((void**)&d_collisionsStale, sizeof(bool));

    // VcudaMalloc((void **)&d_collisions, sizeof(TI_vector<TI_Collision *>));
    // VcudaMemcpyAsync(d_collisions, &h_collisions, sizeof(TI_vector<TI_Collision *>), VcudaMemcpyHostToDevice, stream);

}

void VX3_VoxelyzeKernel::cleanup() {
    //The reason not use ~VX3_VoxelyzeKernel is that will be automatically call multiple times after we use memcpy to clone objects.
    MycudaFree(d_linkMats);
    MycudaFree(d_voxels);
    MycudaFree(d_links);
    MycudaFree(d_collisionsStale);
    // MycudaFree(d_collisions);
}

TI_MaterialLink * VX3_VoxelyzeKernel::getMaterialLink(CVX_MaterialLink* vx_mats) {
    for (int i=0;i<num_d_linkMats;i++) {
        if (h_linkMats[i] == vx_mats) {
            return &d_linkMats[i];
        }
    }
    printf("ERROR: Cannot find the right link material. h_linkMats.size() %ld.\n", h_linkMats.size());
    return NULL;
}

/* Cuda methods : cannot use any CVX_xxx, and no std::, no boost::, and no filesystem. */

__device__ void VX3_VoxelyzeKernel::syncVectors() {
    for (int i=0;i<num_d_linkMats;i++) {
        d_linkMats[i].syncVectors();
    }
}
__device__ bool VX3_VoxelyzeKernel::StopConditionMet(void) //have we met the stop condition yet?
{
    if (StopConditionType!=SC_MAX_SIM_TIME) {
        printf(COLORCODE_BOLD_RED "Only support this type of stop condition for now.\n" COLORCODE_RESET);
        return true;
    }
    return currentTime > StopConditionValue ? true : false;
}

__device__ double VX3_VoxelyzeKernel::recommendedTimeStep() {
    //find the largest natural frequency (sqrt(k/m)) that anything in the simulation will experience, then multiply by 2*pi and invert to get the optimally largest timestep that should retain stability
	double MaxFreq2 = 0.0f; //maximum frequency in the simulation in rad/sec
    for (int i=0;i<num_d_links;i++) {
        TI_Link* pL = d_links+i;
		//axial
		double m1 = pL->pVNeg->mat->mass(),  m2 = pL->pVPos->mat->mass();
		double thisMaxFreq2 = pL->axialStiffness()/(m1<m2?m1:m2);
		if (thisMaxFreq2 > MaxFreq2) MaxFreq2 = thisMaxFreq2;
		//rotational will always be less than or equal
	}
	if (MaxFreq2 <= 0.0f){ //didn't find anything (i.e no links) check for individual voxelss
		for (int i=0;i<num_d_voxels;i++){ //for each link
			double thisMaxFreq2 = d_voxels[i].mat->youngsModulus() * d_voxels[i].mat->nomSize / d_voxels[i].mat->mass(); 
			if (thisMaxFreq2 > MaxFreq2) MaxFreq2 = thisMaxFreq2;
		}
	}
	if (MaxFreq2 <= 0.0f) return 0.0f;
	else return 1.0f/(6.283185f*sqrt(MaxFreq2)); //the optimal timestep is to advance one radian of the highest natural frequency
}

__device__ void VX3_VoxelyzeKernel::updateTemperature() {
    //updates the temperatures For Actuation!
    // different temperatures in different objs are not support for now.
    if (VaryTempEnabled){
		if (TempPeriod > 0) {
            currentTemperature = TempBase + TempAmplitude*sin(2*3.1415926/TempPeriod* currentTime);	//update the global temperature
            int blockSize = 512;
            int gridSize_voxels = (num_d_voxels + blockSize - 1) / blockSize; 
            int blockSize_voxels = num_d_voxels<blockSize ? num_d_voxels : blockSize;
            gpu_update_temperature<<<gridSize_voxels, blockSize_voxels>>>(d_voxels, num_d_voxels, currentTemperature - TempBase);
            cudaDeviceSynchronize();        
        }
	}
}

__device__ bool VX3_VoxelyzeKernel::doTimeStep(float dt) {
    updateTemperature();
    CurStepCount++;
	if (dt==0) return true;
	else if (dt<0) {
        if (!OptimalDt) {
            OptimalDt = recommendedTimeStep();
        }
        if (OptimalDt<1e-10) {
            CUDA_DEBUG_LINE("recommendedTimeStep is zero.");
            return false;
        }
        dt = DtFrac*OptimalDt;
    }
    bool Diverged = false;

    int blockSize = 512;
    int gridSize_links = (num_d_links + blockSize - 1) / blockSize; 
    int blockSize_links = num_d_links<blockSize ? num_d_links : blockSize;
    gpu_update_force<<<gridSize_links, blockSize_links>>>(d_links, num_d_links);
    cudaDeviceSynchronize();

    for (int i = 0; i<num_d_links; i++){
        if (d_links[i].axialStrain() > 100){
            CUDA_DEBUG_LINE("Diverged.");
            Diverged = true; //catch divergent condition! (if any thread sets true we will fail, so don't need mutex...
        }
    }
    if (Diverged) return false;

    // 	if (collisions) updateCollisions();

    int gridSize_voxels = (num_d_voxels + blockSize - 1) / blockSize; 
    int blockSize_voxels = num_d_voxels<blockSize ? num_d_voxels : blockSize;
    gpu_update_voxel<<<gridSize_voxels, blockSize_voxels>>>(d_voxels, num_d_voxels, dt);
    cudaDeviceSynchronize();

    currentTime += dt;
    return true;
}

__device__ void VX3_VoxelyzeKernel::updateCurrentCenterOfMass() {
	double TotalMass = 0;
	TI_Vec3D<> Sum(0,0,0);
	for (int i=0; i<num_d_voxels; i++){
        double ThisMass = d_voxels[i].material()->mass();
		Sum += d_voxels[i].position()*ThisMass;
        TotalMass += ThisMass;
	}

	currentCenterOfMass = Sum/TotalMass;
}

/* Sub GPU Threads */
__global__ void gpu_update_force(TI_Link* links, int num) {
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    if (gindex < num) {
        TI_Link* t = &links[gindex];
        t->updateForces();
        if (t->axialStrain() > 100) { printf("ERROR: Diverged."); }
    }
}
__global__ void gpu_update_voxel(TI_Voxel* voxels, int num, double dt) {
    int gindex = threadIdx.x + blockIdx.x * blockDim.x; 
    if (gindex < num) {
        TI_Voxel* t = &voxels[gindex];
        t->timeStep(dt);
    }
}

__global__ void gpu_update_temperature(TI_Voxel* voxels, int num, double temperature) {
    int gindex = threadIdx.x + blockIdx.x * blockDim.x; 
    if (gindex < num) {
        TI_Voxel* t = &voxels[gindex];
        t->setTemperature(temperature);
    }
}