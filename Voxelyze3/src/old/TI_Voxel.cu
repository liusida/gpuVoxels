#include <vector>
#include "TI_Voxel.h"
#include "TI_VoxelyzeKernel.h"
#include "VX3_VoxelyzeKernel.cuh"
#include "VX3_MemoryCleaner.h"

TI_Voxel::TI_Voxel(CVX_Voxel *p, VX3_VoxelyzeKernel* k): 
ix(p->ix), iy(p->iy), iz(p->iz),
pos(p->pos), linMom(p->linMom), orient(p->orient), angMom(p->angMom),
boolStates(p->boolStates), tempe(p->temp), pStrain(p->pStrain), poissonsStrainInvalid(p->poissonsStrainInvalid),
previousDt(p->previousDt) {
	_voxel = p;
    _kernel = k;

	VcudaMalloc((void **) &mat, sizeof(TI_MaterialVoxel));
	TI_MaterialVoxel temp1(p->mat);
	
	VcudaMemcpy(mat, &temp1, sizeof(TI_MaterialVoxel), VcudaMemcpyHostToDevice);

	for (unsigned i=0;i<6;i++) {
		if (p->links[i]) {
			links[i] = getDevPtrFromHostPtr(p->links[i]);
		} else {
			links[i] = NULL;
		}
	}

	// mat = new TI_MaterialVoxel(p->mat);
	if (p->ext) {
		VcudaMalloc((void **) &ext, sizeof(TI_External));
		TI_External temp2(p->ext);
		VcudaMemcpy(ext, &temp2, sizeof(TI_External), VcudaMemcpyHostToDevice);
	} else {
		ext = NULL;
	}

	//lastColWatchPosition(*p->lastColWatchPosition),colWatch(p->colWatch), nearby(p->nearby)
	// if (p->lastColWatchPosition) {
	// 	lastColWatchPosition = (*p->lastColWatchPosition);
	// }
	// if (p->colWatch) {
	// 	//colWatch = (*p->colWatch);
		
	// }
	// if (p->nearby) {
	// 	//nearby = (*p->nearby);
		
	// }
}

TI_Voxel::~TI_Voxel() {
	if (mat) {
		MycudaFree(mat);
		mat = NULL;
	}
	if (ext) {
		MycudaFree(ext);
		ext = NULL;
	}
}

TI_Link* TI_Voxel::getDevPtrFromHostPtr(CVX_Link* p) {
    //search host pointer in _kernel->h_voxels, get the index and get GPU pointer from _kernel->d_voxels.
	std::vector<CVX_Link *>::iterator it;
    it = find (_kernel->h_links.begin(), _kernel->h_links.end(), p);
    if (it != _kernel->h_links.end()) {
        int index = std::distance(_kernel->h_links.begin(), it);
        return &_kernel->d_links[index];
    }
    else {
        printf("ERROR: link for voxel not found. Maybe the input CVoxelyze* Vx is broken.\n");
    }
    return NULL;
}

CUDA_DEVICE TI_Voxel* TI_Voxel::adjacentVoxel(linkDirection direction) const
{
	TI_Link* pL = links[(int)direction];
	if (pL) return pL->voxel(true)==this ? pL->voxel(false) : pL->voxel(true);
	else return NULL;
}

CUDA_DEVICE void TI_Voxel::addLinkInfo(linkDirection direction, TI_Link* link)
{
	links[direction] = link;
	updateSurface();
}

CUDA_DEVICE void TI_Voxel::removeLinkInfo(linkDirection direction)
{
	links[direction]=NULL;
	updateSurface();
}


CUDA_DEVICE void TI_Voxel::replaceMaterial(TI_MaterialVoxel* newMaterial)
{
	if (newMaterial != NULL){

		linMom *= newMaterial->_mass/mat->_mass; //adjust momentums to keep velocity constant across material change
		angMom *= newMaterial->_momentInertia/mat->_momentInertia;
		setFloorStaticFriction(false);
		poissonsStrainInvalid = true;

		mat = newMaterial;

	}
}

CUDA_DEVICE bool TI_Voxel::isYielded() const
{
	for (int i=0; i<6; i++){
		if (links[i] && links[i]->isYielded()) return true;
	}
	return false;
}

CUDA_DEVICE bool TI_Voxel::isFailed() const
{
	for (int i=0; i<6; i++){
		if (links[i] && links[i]->isFailed()) return true;
	}
	return false;
}

CUDA_DEVICE void TI_Voxel::setTemperature(float temperature)
{
	tempe = temperature;
	for (int i=0; i<6; i++){
		if (links[i] != NULL) links[i]->updateRestLength();
	}
} 


CUDA_DEVICE TI_Vec3D<float> TI_Voxel::externalForce()
{
	TI_Vec3D<float> returnForce(ext->force());
	if (ext->isFixed(X_TRANSLATE) || ext->isFixed(Y_TRANSLATE) || ext->isFixed(Z_TRANSLATE)){
		TI_Vec3D<float> thisForce = (TI_Vec3D<float>) -force();
		if (ext->isFixed(X_TRANSLATE)) returnForce.x = thisForce.x;
		if (ext->isFixed(Y_TRANSLATE)) returnForce.y = thisForce.y;
		if (ext->isFixed(Z_TRANSLATE)) returnForce.z = thisForce.z;
	}
	return returnForce;
}

CUDA_DEVICE TI_Vec3D<float> TI_Voxel::externalMoment()
{
	TI_Vec3D<float> returnMoment(ext->moment());
	if (ext->isFixed(X_ROTATE) || ext->isFixed(Y_ROTATE) || ext->isFixed(Z_ROTATE)){
		TI_Vec3D<float> thisMoment = (TI_Vec3D<float>) -moment();
		if (ext->isFixed(X_ROTATE)) returnMoment.x = thisMoment.x;
		if (ext->isFixed(Y_ROTATE)) returnMoment.y = thisMoment.y;
		if (ext->isFixed(Z_ROTATE)) returnMoment.z = thisMoment.z;
	}
	return returnMoment;
}

CUDA_DEVICE TI_Vec3D<float> TI_Voxel::cornerPosition(voxelCorner corner) const
{
	return (TI_Vec3D<float>)pos + orient.RotateVec3D(cornerOffset(corner));
}

CUDA_DEVICE TI_Vec3D<float> TI_Voxel::cornerOffset(voxelCorner corner) const
{
	TI_Vec3D<> strains;
	for (int i=0; i<3; i++){
		bool posLink = corner&(1<<(2-i))?true:false;
		TI_Link* pL = links[2*i + (posLink?0:1)];
		if (pL && !pL->isFailed()){
			strains[i] = (1 + pL->axialStrain(posLink))*(posLink?1:-1);
		}
		else strains[i] = posLink?1.0:-1.0;
	}

	return (0.5*baseSize()).Scale(strains);
}

//http://klas-physics.googlecode.com/svn/trunk/src/general/Integrator.cpp (reference)
CUDA_DEVICE void TI_Voxel::timeStep(float dt)
{
	previousDt = dt;
	if (dt == 0.0f) return;

	if (ext && ext->isFixedAll()){
		
		pos = originalPosition() + ext->translation();
		orient = ext->rotationQuat();
		haltMotion();
		return;
	}
	
	//Translation
	TI_Vec3D<double> curForce = force();
	
	TI_Vec3D<double> fricForce = curForce;

	if (isFloorEnabled()) {
		floorForce(dt, &curForce); //floor force needs dt to calculate threshold to "stop" a slow voxel into static friction.
	}
	fricForce = curForce - fricForce;

	assert(!(curForce.x != curForce.x) || !(curForce.y != curForce.y) || !(curForce.z != curForce.z)); //assert non QNAN
	linMom += curForce*dt;
	
	TI_Vec3D<double> translate(linMom*(dt*mat->_massInverse)); //movement of the voxel this timestep
	
//	we need to check for friction conditions here (after calculating the translation) and stop things accordingly
	if (isFloorEnabled() && floorPenetration() >= 0){ //we must catch a slowing voxel here since it all boils down to needing access to the dt of this timestep.
		
		double work = fricForce.x*translate.x + fricForce.y*translate.y; //F dot disp
		double hKe = 0.5*mat->_massInverse*(linMom.x*linMom.x + linMom.y*linMom.y); //horizontal kinetic energy
		if(hKe + work <= 0) setFloorStaticFriction(true); //this checks for a change of direction according to the work-energy principle

		if (isFloorStaticFriction()){ //if we're in a state of static friction, zero out all horizontal motion
			linMom.x = linMom.y = 0;
			translate.x = translate.y = 0;
		}
	}
	else setFloorStaticFriction(false);
	

	pos += translate;

	//Rotation
	TI_Vec3D<> curMoment = moment();
	angMom += curMoment*dt;
	
	orient = TI_Quat3D<>(angMom*(dt*mat->_momentInertiaInverse))*orient; //update the orientation
	if (ext){
		double size = mat->nominalSize();
		if (ext->isFixed(X_TRANSLATE)) {pos.x = ix*size + ext->translation().x; linMom.x=0;}
		if (ext->isFixed(Y_TRANSLATE)) {pos.y = iy*size + ext->translation().y; linMom.y=0;}
		if (ext->isFixed(Z_TRANSLATE)) {pos.z = iz*size + ext->translation().z; linMom.z=0;}
		if (ext->isFixedAnyRotation()){ //if any rotation fixed, all are fixed
			if (ext->isFixedAllRotation()){
				orient = ext->rotationQuat();
				angMom = TI_Vec3D<double>();
			}
			else { //partial fixes: slow!
				TI_Vec3D<double> tmpRotVec = orient.ToRotationVector();
				if (ext->isFixed(X_ROTATE)){ tmpRotVec.x=0; angMom.x=0;}
				if (ext->isFixed(Y_ROTATE)){ tmpRotVec.y=0; angMom.y=0;}
				if (ext->isFixed(Z_ROTATE)){ tmpRotVec.z=0; angMom.z=0;}
				orient.FromRotationVector(tmpRotVec);
			}
		}
	}

	
	poissonsStrainInvalid = true;
}
CUDA_DEVICE TI_Vec3D<double> TI_Voxel::force()
{
	
	//forces from internal bonds
	TI_Vec3D<double> totalForce(0,0,0);
	for (int i=0; i<6; i++){ 
		if (links[i]) totalForce += links[i]->force(isNegative((linkDirection)i)); //total force in LCS
	}
	
	totalForce = orient.RotateVec3D(totalForce); //from local to global coordinates
	assert(!(totalForce.x != totalForce.x) || !(totalForce.y != totalForce.y) || !(totalForce.z != totalForce.z)); //assert non QNAN
	
	//other forces
	if (externalExists()) totalForce += external()->force(); //external forces
	totalForce -= velocity()*mat->globalDampingTranslateC(); //global damping f-cv
	totalForce.z += mat->gravityForce(); //gravity, according to f=mg
	
	//no collision yet
	// if (isCollisionsEnabled()){
	// 	for (int i=0;i<colWatch.size();i++){
	// 		totalForce -= colWatch[i]->contactForce(this);
	// 	}
	// }
	return totalForce;
}

CUDA_DEVICE TI_Vec3D<double> TI_Voxel::moment()
{
	//moments from internal bonds
	TI_Vec3D<double> totalMoment(0,0,0);
	for (int i=0; i<6; i++){ 
		if (links[i]) {
			totalMoment += links[i]->moment(isNegative((linkDirection)i)); //total force in LCS		
		}
	}
	totalMoment = orient.RotateVec3D(totalMoment);

	//other moments
	if (externalExists()) totalMoment += external()->moment(); //external moments
	totalMoment -= angularVelocity()*mat->globalDampingRotateC(); //global damping
	return totalMoment;
}


CUDA_DEVICE void TI_Voxel::floorForce(float dt, TI_Vec3D<double>* pTotalForce)
{
	float CurPenetration = floorPenetration(); //for now use the average.
	if (CurPenetration>=0){ 
		TI_Vec3D<double> vel = velocity();
		TI_Vec3D<double> horizontalVel(vel.x, vel.y, 0);
		
		float normalForce = mat->penetrationStiffness()*CurPenetration;

		pTotalForce->z += normalForce - mat->collisionDampingTranslateC()*vel.z; //in the z direction: k*x-C*v - spring and damping


		if (isFloorStaticFriction()){ //If this voxel is currently in static friction mode (no lateral motion) 
			assert(horizontalVel.Length2() == 0);
			float surfaceForceSq = (float)(pTotalForce->x*pTotalForce->x + pTotalForce->y*pTotalForce->y); //use squares to avoid a square root
			float frictionForceSq = (mat->muStatic*normalForce)*(mat->muStatic*normalForce);
		
			if (surfaceForceSq > frictionForceSq) setFloorStaticFriction(false); //if we're breaking static friction, leave the forces as they currently have been calculated to initiate motion this time step
		}
		else { //even if we just transitioned don't process here or else with a complete lack of momentum it'll just go back to static friction
			*pTotalForce -=  mat->muKinetic*normalForce*horizontalVel.Normalized(); //add a friction force opposing velocity according to the normal force and the kinetic coefficient of friction
		}
	}
	else setFloorStaticFriction(false);

}

CUDA_DEVICE TI_Vec3D<float> TI_Voxel::strain(bool poissonsStrain) const
{
	//if no connections in the positive and negative directions of a particular axis, strain is zero
	//if one connection in positive or negative direction of a particular axis, strain is that strain - ?? and force or constraint?
	//if connections in both the positive and negative directions of a particular axis, strain is the average. 
	
	TI_Vec3D<float> intStrRet(0,0,0); //intermediate strain return value. axes according to linkAxis enum
	int numBondAxis[3] = {0}; //number of bonds in this axis (0,1,2). axes according to linkAxis enum
	bool tension[3] = {false};
	for (int i=0; i<6; i++){ //cycle through link directions
		if (links[i]){
			int axis = toAxis((linkDirection)i);
			intStrRet[axis] += links[i]->axialStrain(isNegative((linkDirection)i));
			numBondAxis[axis]++;
		}
	}
	for (int i=0; i<3; i++){ //cycle through axes
		if (numBondAxis[i]==2) intStrRet[i] *= 0.5f; //average
		if (poissonsStrain){
			tension[i] = ((numBondAxis[i]==2) || (ext && (numBondAxis[i]==1 && (ext->isFixed((dofComponent)(1<<i)) || ext->force()[i] != 0)))); //if both sides pulling, or just one side and a fixed or forced voxel...
		}

	}

	if (poissonsStrain){
		if (!(tension[0] && tension[1] && tension[2])){ //if at least one isn't in tension
			float add = 0;
			for (int i=0; i<3; i++) if (tension[i]) add+=intStrRet[i];
			float value = pow( 1.0f + add, -mat->poissonsRatio()) - 1.0f;
			for (int i=0; i<3; i++) if (!tension[i]) intStrRet[i]=value;
		}
	}

	return intStrRet;
}

CUDA_DEVICE TI_Vec3D<float> TI_Voxel::poissonsStrain()
{
	if (poissonsStrainInvalid){
		pStrain = strain(true);
		poissonsStrainInvalid = false;
	}
	return pStrain;
}


CUDA_DEVICE float TI_Voxel::transverseStrainSum(linkAxis axis)
{
	if (mat->poissonsRatio() == 0) return 0;
	
	TI_Vec3D<float> psVec = poissonsStrain();

	switch (axis){
	case X_AXIS: return psVec.y+psVec.z;
	case Y_AXIS: return psVec.x+psVec.z;
	case Z_AXIS: return psVec.x+psVec.y;
	default: return 0.0f;
	}

}

CUDA_DEVICE float TI_Voxel::transverseArea(linkAxis axis)
{
	float size = (float)mat->nominalSize();
	if (mat->poissonsRatio() == 0) return size*size;

	TI_Vec3D<> psVec = poissonsStrain();

	switch (axis){
	case X_AXIS: return (float)(size*size*(1+psVec.y)*(1+psVec.z));
	case Y_AXIS: return (float)(size*size*(1+psVec.x)*(1+psVec.z));
	case Z_AXIS: return (float)(size*size*(1+psVec.x)*(1+psVec.y));
	default: return size*size;
	}
}

CUDA_DEVICE void TI_Voxel::updateSurface()
{
	bool interior = true;
	for (int i=0; i<6; i++) if (!links[i]) interior = false;
	interior ? boolStates |= SURFACE : boolStates &= ~SURFACE;
}


CUDA_DEVICE void TI_Voxel::enableCollisions(bool enabled, float watchRadius) {
	enabled ? boolStates |= COLLISIONS_ENABLED : boolStates &= ~COLLISIONS_ENABLED;
}


CUDA_DEVICE void TI_Voxel::generateNearby(int linkDepth, int gindex, bool surfaceOnly){
	assert(false); //not used.
	// TI_vector<TI_Voxel*> allNearby;
	// allNearby.push_back(this);
	// int iCurrent = 0;
	// for (int k=0; k<linkDepth; k++){
	// 	int iPassEnd = allNearby.size();

	// 	while (iCurrent != iPassEnd){
	// 		TI_Voxel* pV = allNearby[iCurrent++];
			
	// 		for (int i=0; i<6; i++){
	// 			printf("pV %p gindex %d \n", pV, gindex);
	// 			TI_Voxel* pV2 = pV->adjacentVoxel((linkDirection)i);
	// 			//if (pV2 && std::find(allNearby.begin(), allNearby.end(), pV2) == allNearby.end()) allNearby.push_back(pV2);
	// 			if (pV2) {
	// 				bool finded = false;
	// 				for (unsigned j=0;j<allNearby.size();j++) {
	// 					if (pV2==allNearby[j]) {
	// 						finded = true;
	// 						break;
	// 					}
	// 				}
	// 				if (!finded) {
	// 					if (gindex==1) {
	// 						printf("pV2 %p\n", pV2);
	// 					}
	// 						allNearby.push_back(pV2, true);
	// 					if (gindex==1) {
	// 						for (int k=0;k<allNearby.size();k++)
	// 							printf("gindex %d (%p)allNearby[%d] %p\n", gindex, &allNearby, k, allNearby[k]);
	// 					}
	// 					else
	// 						allNearby.push_back(pV2, false);
	// 				}
	// 			}

	// 		}
	// 	}
	// }
	// printf("ok.\n");

	// nearby.clear();
	// for (unsigned i=0;i<allNearby.size();i++) {
	// 	TI_Voxel* pV = allNearby[i];
	// 	if (pV->isSurface() && pV != this) nearby.push_back(pV);		
	// }
}
