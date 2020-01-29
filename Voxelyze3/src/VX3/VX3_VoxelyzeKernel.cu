#include "VX3_VoxelyzeKernel.cuh"
#include "VX3_MemoryCleaner.h"

/* Sub GPU Threads */
__global__ void gpu_update_force(VX3_Link** links, int num);
__global__ void gpu_update_voxel(VX3_Voxel* voxels, int num, double dt);
__global__ void gpu_update_temperature(VX3_Voxel* voxels, int num, double TempAmplitude, double TempPeriod, double currentTime);
__global__ void gpu_update_attach(VX3_Voxel** surface_voxels, int num, double watchDistance, VX3_VoxelyzeKernel* k);
/* Host methods */

VX3_VoxelyzeKernel::VX3_VoxelyzeKernel(CVX_Sim* In) {

    voxSize = In->Vx.voxSize;
    
    num_d_voxelMats = In->Vx.voxelMats.size();
    VcudaMalloc((void **)&d_voxelMats, num_d_voxelMats * sizeof(VX3_MaterialVoxel));
    {
        int i=0;
        for (auto mat: In->Vx.voxelMats) {
            VX3_MaterialVoxel tmp_voxelMat( mat, this );
            VcudaMemcpy( d_voxelMats+i, &tmp_voxelMat, sizeof(VX3_MaterialVoxel), VcudaMemcpyHostToDevice );
            h_voxelMats.push_back(mat);
            i++;
        }
    }

    num_d_linkMats = In->Vx.linkMats.size();
    VcudaMalloc( (void **)&d_linkMats, num_d_linkMats * sizeof(VX3_MaterialLink));
    {
        int i = 0;
        std::vector<VX3_MaterialLink*> tmp_v_linkMats;
        for (CVX_MaterialLink* mat:In->Vx.linkMats) {
            // printf("mat->vox1Mat %p, mat->vox2Mat %p.\n", mat->vox1Mat, mat->vox2Mat);
            VX3_MaterialLink tmp_linkMat( mat, this );
            VcudaMemcpy( d_linkMats+i, &tmp_linkMat, sizeof(VX3_MaterialLink), VcudaMemcpyHostToDevice );
            tmp_v_linkMats.push_back(d_linkMats+i);
            h_linkMats.push_back( mat );
            i++;
        }
        hd_v_linkMats = VX3_hdVector<VX3_MaterialLink*>(tmp_v_linkMats);
    }

    num_d_voxels = In->Vx.voxelsList.size();
    VcudaMalloc( (void **)&d_voxels, num_d_voxels * sizeof(VX3_Voxel));
    for (int i=0;i<num_d_voxels;i++) {
        h_voxels.push_back( In->Vx.voxelsList[i] );
    }

    num_d_links = In->Vx.linksList.size();
    std::vector<VX3_Link*> tmp_v_links;
    VcudaMalloc( (void **)&d_links, num_d_links * sizeof(VX3_Link));
    VX3_Link* tmp_link_cache = (VX3_Link*) malloc(num_d_links * sizeof(VX3_Link));
    for (int i=0;i<num_d_links;i++) {
        VX3_Link tmp_link( In->Vx.linksList[i], this );
        memcpy(tmp_link_cache+i, &tmp_link, sizeof(VX3_Link));
        tmp_v_links.push_back(d_links+i); //not copied yet, but still ok to get the address
        h_links.push_back( In->Vx.linksList[i] );
    }
    VcudaMemcpy( d_links, tmp_link_cache, num_d_links * sizeof(VX3_Link), VcudaMemcpyHostToDevice );
    hd_v_links = VX3_hdVector<VX3_Link*>(tmp_v_links);

    for (int i=0;i<num_d_voxels;i++) {
        //set values for GPU memory space
        VX3_Voxel tmp_voxel(In->Vx.voxelsList[i], this);
        VcudaMemcpy(d_voxels+i, &tmp_voxel, sizeof(VX3_Voxel), VcudaMemcpyHostToDevice);
    }

    //Not all data is in Vx, here are others:
    DtFrac = In->DtFrac;
    StopConditionType = In->StopConditionType;
    StopConditionValue = In->StopConditionValue;
    TempEnabled = In->pEnv->TempEnabled;
    VaryTempEnabled = In->pEnv->VaryTempEnabled;
    TempBase = In->pEnv->TempBase;
    TempAmplitude = In->pEnv->TempAmplitude;
    TempPeriod = In->pEnv->TempPeriod;
    // currentTemperature = TempBase + TempAmplitude;

    d_surface_voxels = NULL;
}

void VX3_VoxelyzeKernel::cleanup() {
    //The reason not use ~VX3_VoxelyzeKernel is that will be automatically call multiple times after we use memcpy to clone objects.
    MycudaFree(d_linkMats);
    MycudaFree(d_voxels);
    MycudaFree(d_links);
    MycudaFree(d_collisionsStale);
    if (d_surface_voxels) {
        MycudaFree(d_surface_voxels); //can __device__ malloc pointer be freed by cudaFree in __host__??
    }
    // MycudaFree(d_collisions);
}

/* Cuda methods : cannot use any CVX_xxx, and no std::, no boost::, and no filesystem. */

__device__ void VX3_VoxelyzeKernel::syncVectors() {
    d_v_linkMats.clear();
    for (int i=0;i<hd_v_linkMats.size();i++) {
        d_v_linkMats.push_back(hd_v_linkMats[i]);
    }

    d_v_links.clear();
    for (int i=0;i<hd_v_links.size();i++) {
        d_v_links.push_back(hd_v_links[i]);
    }

    for (int i=0;i<num_d_voxelMats;i++) {
        d_voxelMats[i].syncVectors();
    }
    
    for (int i=0;i<num_d_linkMats;i++) {
        d_linkMats[i].syncVectors();
    }
}
__device__ bool VX3_VoxelyzeKernel::StopConditionMet(void) //have we met the stop condition yet?
{
    if (StopConditionType!=SC_MAX_SIM_TIME) {
        printf(COLORCODE_BOLD_RED "StopConditionType: %d. Type of stop condition no supported for now.\n" COLORCODE_RESET, StopConditionType);
        return true;
    }
    if (forceExit) return true;
    return currentTime > StopConditionValue ? true : false;
}

__device__ double VX3_VoxelyzeKernel::recommendedTimeStep() {
    //find the largest natural frequency (sqrt(k/m)) that anything in the simulation will experience, then multiply by 2*pi and invert to get the optimally largest timestep that should retain stability
	double MaxFreq2 = 0.0f; //maximum frequency in the simulation in rad/sec
    for (int i=0;i<num_d_links;i++) {
        VX3_Link* pL = d_links+i;
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
            int blockSize = 512;
            int minGridSize;
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, gpu_update_temperature, 0, num_d_voxels); //Dynamically calculate blockSize
            int gridSize_voxels = (num_d_voxels + blockSize - 1) / blockSize; 
            int blockSize_voxels = num_d_voxels<blockSize ? num_d_voxels : blockSize;
            gpu_update_temperature<<<gridSize_voxels, blockSize_voxels>>>(d_voxels, num_d_voxels, TempAmplitude, TempPeriod, currentTime);
            CUDA_CHECK_AFTER_CALL();
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

    int blockSize;
    int minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, gpu_update_force, 0, d_v_links.size()); //Dynamically calculate blockSize
    int gridSize_links = (d_v_links.size() + blockSize - 1) / blockSize; 
    int blockSize_links = d_v_links.size()<blockSize ? d_v_links.size() : blockSize;
    // printf("gpu_update_force<<<%d,%d>>>(...,%d);\n", gridSize_links, blockSize_links, d_v_links.size());
    gpu_update_force<<<gridSize_links, blockSize_links>>>(&d_v_links[0], d_v_links.size());
    CUDA_CHECK_AFTER_CALL();
    cudaDeviceSynchronize();

    for (int i = 0; i<d_v_links.size(); i++){
        if (d_v_links[i]->axialStrain() > 100){
            CUDA_DEBUG_LINE("Diverged.");
            Diverged = true; //catch divergent condition! (if any thread sets true we will fail, so don't need mutex...
        }
    }
    if (Diverged) return false;

    if (enableAttach) updateAttach();

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, gpu_update_voxel, 0, num_d_voxels); //Dynamically calculate blockSize
    int gridSize_voxels = (num_d_voxels + blockSize - 1) / blockSize; 
    int blockSize_voxels = num_d_voxels<blockSize ? num_d_voxels : blockSize;
    gpu_update_voxel<<<gridSize_voxels, blockSize_voxels>>>(d_voxels, num_d_voxels, dt);
    CUDA_CHECK_AFTER_CALL();
    cudaDeviceSynchronize();

    currentTime += dt;
    return true;
}

__device__ void VX3_VoxelyzeKernel::updateAttach()
{
    //for each surface voxel pair, check distance < watchDistance, make a new link between these two voxels, updateSurface().
    int blockSize = 32;
    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid((num_d_surface_voxels + dimBlock.x - 1) / dimBlock.x, (num_d_surface_voxels + dimBlock.y - 1) / dimBlock.y);
    gpu_update_attach<<<dimGrid, dimBlock>>>(d_surface_voxels, num_d_surface_voxels, watchDistance, this); //invoke two dimensional gpu threads 'CUDA C++ Programming Guide', Nov 2019, P52.
    CUDA_CHECK_AFTER_CALL();
}


__device__ void VX3_VoxelyzeKernel::updateCurrentCenterOfMass() {
	double TotalMass = 0;
	VX3_Vec3D<> Sum(0,0,0);
	for (int i=0; i<num_d_voxels; i++){
        double ThisMass = d_voxels[i].material()->mass();
		Sum += d_voxels[i].position()*ThisMass;
        TotalMass += ThisMass;
	}

	currentCenterOfMass = Sum/TotalMass;
}

__device__ void VX3_VoxelyzeKernel::regenerateSurfaceVoxels() {
    // regenerate d_surface_voxels
    if (d_surface_voxels) {
        delete d_surface_voxels;
        d_surface_voxels = NULL;
    }
    VX3_dVector<VX3_Voxel*> tmp;
    for (int i=0;i<num_d_voxels;i++) {
        if (d_voxels[i].isSurface()) {
            tmp.push_back(&d_voxels[i]);
        }
    }
    num_d_surface_voxels = tmp.size();
    d_surface_voxels = (VX3_Voxel **)malloc( num_d_surface_voxels * sizeof(VX3_Voxel) );
    for (int i=0;i<num_d_surface_voxels;i++) {
        d_surface_voxels[i] = tmp[i];
    }
}

__device__ VX3_MaterialLink* VX3_VoxelyzeKernel::combinedMaterial(VX3_MaterialVoxel* mat1, VX3_MaterialVoxel* mat2) 
{
    for (int i=0;i<d_v_linkMats.size();i++) {
        VX3_MaterialLink* thisMat = d_v_linkMats[i];
		if ((thisMat->vox1Mat == mat1 && thisMat->vox2Mat == mat2) || (thisMat->vox1Mat == mat2 && thisMat->vox2Mat == mat1))
			return thisMat; //already exist
    }
    
    VX3_MaterialLink* newMat = new VX3_MaterialLink(mat1, mat2); //where to free this?
    d_v_linkMats.push_back(newMat);
	mat1->d_dependentMaterials.push_back(newMat);
	mat2->d_dependentMaterials.push_back(newMat);

	return newMat;
}


/* Sub GPU Threads */
__global__ void gpu_update_force(VX3_Link** links, int num) {
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    if (gindex < num) {
        VX3_Link* t = links[gindex];
        t->updateForces();
        if (t->axialStrain() > 100) { printf("ERROR: Diverged."); }
    }
}
__global__ void gpu_update_voxel(VX3_Voxel* voxels, int num, double dt) {
    int gindex = threadIdx.x + blockIdx.x * blockDim.x; 
    if (gindex < num) {
        VX3_Voxel* t = &voxels[gindex];
        t->timeStep(dt);
    }
}

__global__ void gpu_update_temperature(VX3_Voxel* voxels, int num, double TempAmplitude, double TempPeriod, double currentTime) {
    int gindex = threadIdx.x + blockIdx.x * blockDim.x; 
    if (gindex < num) {
    //vfloat tmp = pEnv->GetTempAmplitude() * sin(2*3.1415926f*(CurTime/pEnv->GetTempPeriod() + pV->phaseOffset)) - pEnv->GetTempBase();
        VX3_Voxel* t = &voxels[gindex];
        double currentTemperature = TempAmplitude*sin(2*3.1415926f*(currentTime/TempPeriod + t->phaseOffset));	//update the global temperature
        t->setTemperature(currentTemperature);
        // t->setTemperature(0.0f);
    }
}
__global__ void gpu_update_attach(VX3_Voxel** surface_voxels, int num, double watchDistance, VX3_VoxelyzeKernel* k) {
    int first = threadIdx.x + blockIdx.x * blockDim.x; 
    int second = threadIdx.y + blockIdx.y * blockDim.y; 
    if (first<num && second<first) {
        VX3_Voxel* voxel1 = surface_voxels[first];
        VX3_Voxel* voxel2 = surface_voxels[second];
        double diffx = voxel1->pos.x - voxel2->pos.x;
        double diffy = voxel1->pos.y - voxel2->pos.y;
        double diffz = voxel1->pos.z - voxel2->pos.z;
        if (diffx>watchDistance || diffx<-watchDistance) return;
        if (diffy>watchDistance || diffy<-watchDistance) return;
        if (diffz>watchDistance || diffz<-watchDistance) return;
        //to exclude voxels already have link between them.
        for (int i=0;i<6;i++) {
            if (voxel1->links[i]) {
                if (voxel1->links[i]->pVNeg == voxel2 || voxel1->links[i]->pVPos == voxel2) return;
            }
        }
        //create a link between voxel1 and voxel2 (orientation matters?)
        VX3_MaterialLink* mat = k->combinedMaterial(voxel1->material(), voxel2->material());
		VX3_Link* pL = new VX3_Link(voxel1, voxel2, mat); //make the new link (change to both materials, etc.
    
        k->d_v_links.push_back(pL);							//add to the list

        printf("hmmm.... %p %p distance=> %f %f %f\n", voxel1, voxel2, diffx, diffy, diffz);
    }
}