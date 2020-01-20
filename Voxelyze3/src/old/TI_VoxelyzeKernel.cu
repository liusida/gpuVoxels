#ifdef _0
#include <iostream>
#include <algorithm>
#include "cuda_occupancy.h"

#include "TI_VoxelyzeKernel.h"
#include "TI_Utils.h"

TI_VoxelyzeKernel::TI_VoxelyzeKernel( CVoxelyze* vx ):
currentTime(vx->currentTime), nearbyStale(true), collisionsStale(true)
{
    _vx = vx;
    
    for (auto mat:vx->linkMats) {
        TI_MaterialLink * d_mat;
        VcudaMalloc((void **) &d_mat, sizeof(TI_MaterialLink));
        TI_MaterialLink temp = TI_MaterialLink(mat);
        VcudaMemcpy(d_mat, &temp, sizeof(TI_MaterialLink), VcudaMemcpyHostToDevice);
        d_linkMats.push_back(d_mat);
        h_linkMats.push_back(mat);
    }
    //for voxel: need to allocate memory first, then set the value, because there are links in voxels and voxels in links.
    for (auto voxel: vx->voxelsList) {
        //alloc a GPU memory space
        TI_Voxel * d_voxel;
        VcudaMalloc((void **) &d_voxel, sizeof(TI_Voxel));
        //save the pointer
        d_voxels.push_back(d_voxel);
        //save host pointer as well
        h_voxels.push_back(voxel);
    }
    for (auto link: vx->linksList) {
        //alloc a GPU memory space
        TI_Link * d_link;
        VcudaMalloc((void **) &d_link, sizeof(TI_Link));
        //set values for GPU memory space
        TI_Link temp = TI_Link(link, this);
        VcudaMemcpy(d_link, &temp, sizeof(TI_Link), VcudaMemcpyHostToDevice);
        //save the pointer
        d_links.push_back(d_link);
        //save host pointer as well
        h_links.push_back(link);
    }
    for (unsigned i=0;i<vx->voxelsList.size();i++) {
        TI_Voxel * d_voxel = d_voxels[i];
        CVX_Voxel * voxel = vx->voxelsList[i];
        //set values for GPU memory space
        TI_Voxel temp(voxel, this);
        VcudaMemcpy(d_voxel, &temp, sizeof(TI_Voxel), VcudaMemcpyHostToDevice);
    }


    VcudaMalloc((void**)&d_collisionsStale, sizeof(bool));

    VcudaMalloc((void **)&d_collisions, sizeof(TI_vector<TI_Collision *>));
    VcudaMemcpy(d_collisions, &h_collisions, sizeof(TI_vector<TI_Collision *>), VcudaMemcpyHostToDevice);
}

TI_VoxelyzeKernel::~TI_VoxelyzeKernel()
{
}

__global__ void gpu_function_1(int* a, int num) {
    int gindex = threadIdx.x + blockIdx.x * blockDim.x; 
    if (gindex < num) {
        a[gindex] = gindex;
    }
}

void TI_VoxelyzeKernel::simpleGPUFunction() {
    int* d_a;
    int* a;
    int num = 10;
    int mem_size = num * sizeof(int);

    a = (int *) malloc(mem_size);
    VcudaMalloc( &d_a, mem_size );

    gpu_function_1<<<1,num>>>(d_a, num);
    VcudaMemcpy(a, d_a, mem_size, VcudaMemcpyDeviceToHost);

    for (int i=0;i<num;i++) {
        std::cout<< a[i] << ",";
    }
    std::cout << std::endl;
}

__global__
void gpu_update_force(TI_Link** links, int num) {
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    if (gindex < num) {
        TI_Link* t = links[gindex];
        t->updateForces();
        if (t->axialStrain() > 100) { printf("ERROR: Diverged."); }
    }
}
__global__
void gpu_update_voxel(TI_Voxel** voxels, int num, double dt) {
    int gindex = threadIdx.x + blockIdx.x * blockDim.x; 
    if (gindex < num) {
        TI_Voxel* t = voxels[gindex];
        t->timeStep(dt);
    }
}
__global__
void generate_voxels_Nearby(TI_Voxel** voxels, int num, float watchRadiusVx) {
    int gindex = threadIdx.x + blockIdx.x * blockDim.x; 
    if (gindex < num) {
        TI_Voxel* t = voxels[gindex];
        t->generateNearby(watchRadiusVx*2, gindex, false);
    }
}

__global__
void gpu_check_moved_far_enough(TI_Voxel** voxels, int num, bool* pCollisionsStale, double recalcDist) {
    int gindex = threadIdx.x + blockIdx.x * blockDim.x; 
    if (gindex < num) {
        TI_Voxel *pV = voxels[gindex];
		if (pV->isSurface() && (pV->pos - pV->lastColWatchPosition).Length2() > recalcDist*recalcDist){
            *pCollisionsStale = true; // far enough
        }            
    }
}
__global__
void gpu_clear_collision_for_voxels(TI_Voxel** voxels, int num) {
    int gindex = threadIdx.x + blockIdx.x * blockDim.x; 
    if (gindex < num) {
        TI_Voxel* t = voxels[gindex];
        t->colWatch.clear();
    }
}
__global__
void gpu_clear_collision(TI_vector<TI_Collision *>* d_collisions) {
    for (unsigned i=0;i<d_collisions->size();i++) {
        TI_Collision *pCol = d_collisions->get(i);
        //debugDev( printf("delete TI_Collision %p", pCol) );
        delete pCol;
    }
    d_collisions->clear();
    // debugDev( printf("d_collisions cleared: %d;\t", d_collisions->size()) );
}
//TODO: TI_vector is not thread-safe. so d_collisions should be change to "Mapper Reducer". 
//not used
// __global__
// void gpu_generate_collision(TI_Voxel** voxels, int num, double threshRadiusSq, TI_vector<TI_Collision *>* d_collisions) {
//     int gindex_x = threadIdx.x + blockIdx.x * blockDim.x; 
//     int gindex_y = threadIdx.y + blockIdx.y * blockDim.y;
//     if (gindex_x<gindex_y && gindex_y<num) {
//         TI_Voxel* pV1 = voxels[gindex_x];
//         TI_Voxel* pV2 = voxels[gindex_y];
//         if (!pV1->isInterior() && !pV2->isInterior()) { //don't care about interior voxels here.
//             if ((pV1->pos - pV2->pos).Length2() <= threshRadiusSq) { //discard anything outside the watch radius
//                 if (!pV1->nearby.find(pV2)) { //discard if in the connected lattice array
//                     TI_Collision *pCol = new TI_Collision(pV1, pV2);
//                     d_collisions->push_back(pCol);
//                     pV1->colWatch.push_back(pCol);
//                     pV2->colWatch.push_back(pCol);
//                     //debugDev( printf("new TI_Collision %p", pCol) );
//                     //debugDev( printf("d_collisions->size() %d",d_collisions->size()))
//                 }
//             }
//         }
//     }
//     if (gindex_x==gindex_y && gindex_y<num) { //update last watch
//         TI_Voxel* pV = voxels[gindex_x];
//         pV->lastColWatchPosition = pV->pos;
//     }
// }
__global__
void gpu_update_contact_force(TI_vector<TI_Collision *>* d_collisions) {
    int gindex = threadIdx.x + blockIdx.x * blockDim.x; 
    if (gindex < d_collisions->size()) {
        // debugDev( printf("d_collisions update force: %d;\t", d_collisions->size()) );
        TI_Collision* t = d_collisions->get(gindex);
        //debugDev( printf("%p hit %p ?", t->pV1, t->pV2) );
        //debugDevice("pV1", t->pV1->pos.debug());
        //debugDevice("pV2", t->pV2->pos.debug());
        t->updateContactForce();
    }
}


void TI_VoxelyzeKernel::clearCollisions() {
    int blockSize = 1024;
    int num_voxels = d_voxels.size();
    int blockSize_voxels = std::min(num_voxels, blockSize);
    int gridSize_voxels = (num_voxels + blockSize - 1) / blockSize; 
    gpu_clear_collision_for_voxels<<<gridSize_voxels, blockSize_voxels>>>(thrust::raw_pointer_cast(d_voxels.data()), num_voxels);
    cudaDeviceSynchronize();
    gpu_clear_collision<<<1,1>>>(d_collisions);
    cudaDeviceSynchronize();
}

void TI_VoxelyzeKernel::regenerateCollisions(double threshRadiusSq) {
    //not used
    // clearCollisions();
    
    // int num_voxels = d_voxels.size();
    // int blockSize = 32; //TODO: How to optimize this?
    // dim3 threadsPerBlock(blockSize, blockSize);
    // dim3 numBlocks( (num_voxels + blockSize - 1) / blockSize, (num_voxels + blockSize - 1) / blockSize );

    // gpu_generate_collision<<<numBlocks, threadsPerBlock>>>(thrust::raw_pointer_cast(d_voxels.data()), num_voxels, threshRadiusSq, d_collisions);
    // cudaDeviceSynchronize();
} //regenerateCollisions

void TI_VoxelyzeKernel::updateCollisions() {
    //not used
    // int blockSize = 1024;
    // int num_voxels = d_voxels.size();
    // int blockSize_voxels = std::min(num_voxels, blockSize);
    // int gridSize_voxels = (num_voxels + blockSize - 1) / blockSize; 
    // double watchRadiusVx = 2*_vx->boundingRadius + _vx->watchDistance; //outer radius to track all voxels within
	// double watchRadiusMm = (double)(_vx->voxSize * watchRadiusVx); //outer radius to track all voxels within
	// double recalcDist = (double)(_vx->voxSize * _vx->watchDistance / 2 ); //if the voxel moves further than this radius, recalc! //1/2 the allowabl, accounting for 0.5x radius of the voxel iself

	// if (nearbyStale){
    //     generate_voxels_Nearby<<<gridSize_voxels, blockSize_voxels>>>(thrust::raw_pointer_cast(d_voxels.data()), num_voxels, watchRadiusVx);
    //     cudaDeviceSynchronize();
	// 	nearbyStale = false;
	// 	collisionsStale = true;
    // }
    
    // //check if any voxels have moved far enough to make collisions stale
    // VcudaMemcpy(d_collisionsStale, &collisionsStale, sizeof(bool), VcudaMemcpyHostToDevice);
    // gpu_check_moved_far_enough<<<gridSize_voxels, blockSize_voxels>>>(thrust::raw_pointer_cast(d_voxels.data()), num_voxels, d_collisionsStale, recalcDist);
    // cudaDeviceSynchronize();
    // VcudaMemcpy(&collisionsStale, d_collisionsStale, sizeof(bool), VcudaMemcpyDeviceToHost);
    
    // if (collisionsStale){
    //     regenerateCollisions(watchRadiusMm*watchRadiusMm);
    //     collisionsStale = false;
    // }
    // VcudaMemcpy(&h_collisions, d_collisions, sizeof(TI_vector<TI_Collision *>), VcudaMemcpyDeviceToHost);

    // //check if any voxels have moved far enough to make collisions stale
    // int num_collisions = h_collisions.size();
    // if (num_collisions) {
    //     int gridSize_collisions = (num_collisions + blockSize - 1) / blockSize; 
    //     int blockSize_collisions = std::min(num_collisions, blockSize);
    //     gpu_update_contact_force<<<gridSize_collisions, blockSize_collisions>>>(d_collisions);
    //     cudaDeviceSynchronize();
    // }
}

CUDA_CALLABLE_MEMBER bool TI_VoxelyzeKernel::doTimeStep(double dt) {
    int blockSize = 1024;
    int num_links = d_links.size();
    int blockSize_links = min(num_links, blockSize);
    int num_voxels = d_voxels.size();
    int blockSize_voxels = min(num_voxels, blockSize);
    int gridSize_links = (num_links + blockSize - 1) / blockSize; 
    int gridSize_voxels = (num_voxels + blockSize - 1) / blockSize;
    gpu_update_force<<<gridSize_links, blockSize_links>>>(thrust::raw_pointer_cast(d_links.data()), num_links);
    cudaDeviceSynchronize();
    
    //TODO: rewrite updateCollisions();

    gpu_update_voxel<<<gridSize_voxels, blockSize_voxels>>>(thrust::raw_pointer_cast(d_voxels.data()), num_voxels, dt);
    cudaDeviceSynchronize();

    currentTime += dt;
    return true;
}

void TI_VoxelyzeKernel::readVoxelsPosFromDev() {
    for (auto l:read_links) delete l;
    for (auto v:read_voxels) delete v;
    read_links.clear();
    read_voxels.clear();

    for (unsigned i=0;i<d_voxels.size();i++) {
        TI_Voxel* temp = (TI_Voxel*) malloc(sizeof(TI_Voxel));
        VcudaMemcpy(temp, d_voxels[i], sizeof(TI_Voxel), VcudaMemcpyDeviceToHost);
        read_voxels.push_back(temp);
    }
    for (unsigned i=0;i<d_links.size();i++) {
        TI_Link* temp = (TI_Link*) malloc(sizeof(TI_Link));
        VcudaMemcpy(temp, d_links[i], sizeof(TI_Link), VcudaMemcpyDeviceToHost);
        read_links.push_back(temp);
    }
}

TI_MaterialLink * TI_VoxelyzeKernel::getMaterialLink(CVX_MaterialLink* vx_mats) {
    for (int i=0;i<h_linkMats.size();i++) {
        if (h_linkMats[i] == vx_mats) {
            return d_linkMats[i];
        }
    }

    printf("ERROR: Cannot find the right link material. h_linkMats.size() %ld.\n", h_linkMats.size());
    return NULL;
}
#endif