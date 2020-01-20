#include <vector>
#include "VX3_VoxelyzeKernel.h"
#include "TI_Link.h"
#include "TI_VoxelyzeKernel.h"
#include "TI_MaterialLink.h"

TI_Link::TI_Link(CVX_Link* p, VX3_VoxelyzeKernel* k) :
forceNeg(p->forceNeg), forcePos(p->forcePos),
momentNeg(p->momentNeg), momentPos(p->momentPos),
strain(p->strain), maxStrain(p->maxStrain), strainOffset(p->strainOffset),
boolStates(p->boolStates), axis(p->axis),
strainRatio(p->strainRatio), 
pos2(p->pos2), angle1v(p->angle1v), angle2v(p->angle2v),
angle1(p->angle1), angle2(p->angle2), smallAngle(p->smallAngle),
currentRestLength(p->currentRestLength), currentTransverseArea(p->currentTransverseArea),
currentTransverseStrainSum(p->currentTransverseStrainSum),
_stress(p->_stress)
{
    _link = p;
    _kernel = k;

    pVNeg = getDevPtrFromHostPtr(p->pVNeg);
    pVPos = getDevPtrFromHostPtr(p->pVPos);

	mat = k->getMaterialLink(p->mat);
}

TI_Voxel* TI_Link::getDevPtrFromHostPtr(CVX_Voxel* p) {
    //search host pointer in _kernel->h_voxels, get the index and get GPU pointer from _kernel->d_voxels.
	std::vector<CVX_Voxel *>::iterator it;
    it = find (_kernel->h_voxels.begin(), _kernel->h_voxels.end(), p);
    if (it != _kernel->h_voxels.end()) {
        int index = std::distance(_kernel->h_voxels.begin(), it);
        return &_kernel->d_voxels[index];
    }
    else {
        printf("ERROR: voxel for link not found. Maybe the input CVoxelyze* Vx is broken.\n");
    }
    return NULL;
}

CUDA_DEVICE void TI_Link::test() {
	
}



CUDA_DEVICE TI_Quat3D<double> TI_Link::orientLink(/*double restLength*/) //updates pos2, angle1, angle2, and smallAngle
{
	pos2 = toAxisX(TI_Vec3D<double>(pVPos->position() - pVNeg->position())); //digit truncation happens here...

	angle1 = toAxisX(pVNeg->orientation());
	angle2 = toAxisX(pVPos->orientation());

	TI_Quat3D<double> totalRot = angle1.Conjugate(); //keep track of the total rotation of this bond (after toAxisX())
	pos2 = totalRot.RotateVec3D(pos2);
	angle2 = totalRot*angle2;
	angle1 = TI_Quat3D<>(); //zero for now...

	//small angle approximation?
	float SmallTurn = (float)((abs(pos2.z)+abs(pos2.y))/pos2.x);
	float ExtendPerc = (float)(abs(1-pos2.x/currentRestLength));
	if (!smallAngle /*&& angle2.IsSmallAngle()*/ && SmallTurn < SA_BOND_BEND_RAD && ExtendPerc < SA_BOND_EXT_PERC){
		smallAngle = true;
		setBoolState(LOCAL_VELOCITY_VALID, false);
	}
	else if (smallAngle && (/*!angle2.IsSmallishAngle() || */SmallTurn > HYSTERESIS_FACTOR*SA_BOND_BEND_RAD || ExtendPerc > HYSTERESIS_FACTOR*SA_BOND_EXT_PERC)){
		smallAngle = false;
		setBoolState(LOCAL_VELOCITY_VALID, false);
	}

	if (smallAngle)	{ //Align so Angle1 is all zeros
		pos2.x -= currentRestLength; //only valid for small angles
	}
	else { //Large angle. Align so that Pos2.y, Pos2.z are zero.
		angle1.FromAngleToPosX(pos2); //get the angle to align Pos2 with the X axis
		totalRot = angle1 * totalRot; //update our total rotation to reflect this
		angle2 = angle1 * angle2; //rotate angle2
		pos2 = TI_Vec3D<>(pos2.Length() - currentRestLength, 0, 0); 
	}

	angle1v = angle1.ToRotationVector();
	angle2v = angle2.ToRotationVector();

	assert(!(angle1v.x != angle1v.x) || !(angle1v.y != angle1v.y) || !(angle1v.z != angle1v.z)); //assert non QNAN
	assert(!(angle2v.x != angle2v.x) || !(angle2v.y != angle2v.y) || !(angle2v.z != angle2v.z)); //assert non QNAN


	return totalRot;
}

CUDA_DEVICE float TI_Link::axialStrain(bool positiveEnd) const
{
	return positiveEnd ? 2.0f*strain*strainRatio/(1.0f+strainRatio) : 2.0f*strain/(1.0f+strainRatio);
}


CUDA_DEVICE bool TI_Link::isYielded() const
{
	return mat->isYielded(maxStrain);
}

CUDA_DEVICE bool TI_Link::isFailed() const
{
	return mat->isFailed(maxStrain);
}

CUDA_DEVICE void TI_Link::updateRestLength()
{
	currentRestLength = 0.5*(pVNeg->baseSize(axis) + pVPos->baseSize(axis));
}

CUDA_DEVICE void TI_Link::updateTransverseInfo()
{
	currentTransverseArea = 0.5f*(pVNeg->transverseArea(axis)+pVPos->transverseArea(axis));
	currentTransverseStrainSum = 0.5f*(pVNeg->transverseStrainSum(axis)+pVPos->transverseStrainSum(axis));

}

CUDA_DEVICE void TI_Link::updateForces()
{
	//time start 0us
	TI_Vec3D<double> oldPos2 = pos2, oldAngle1v = angle1v, oldAngle2v = angle2v; //remember the positions/angles from last timestep to calculate velocity

	orientLink(/*restLength*/); //sets pos2, angle1, angle2
	//time 87.876us
	TI_Vec3D<double> dPos2 = 0.5*(pos2-oldPos2); //deltas for local damping. velocity at center is half the total velocity
	TI_Vec3D<double> dAngle1 = 0.5*(angle1v-oldAngle1v);
	TI_Vec3D<double> dAngle2 = 0.5*(angle2v-oldAngle2v);
	//time 87.651us
	//if volume effects..
	if (!mat->isXyzIndependent() || currentTransverseStrainSum != 0) { //currentTransverseStrainSum != 0 catches when we disable poissons mid-simulation
		//updateTransverseInfo(); 
	}	
	//time 110.08us
	_stress = updateStrain((float)(pos2.x/currentRestLength));
	//time 119.13us
	if (isFailed()){forceNeg = forcePos = momentNeg = momentPos = TI_Vec3D<double>(0,0,0); return;}
	//time 120.48us
	float b1=mat->_b1, b2=mat->_b2, b3=mat->_b3, a2=mat->_a2; //local copies
	//Beam equations. All relevant terms are here, even though some are zero for small angle and others are zero for large angle (profiled as negligible performance penalty)
	forceNeg = TI_Vec3D<double> (	_stress*currentTransverseArea, //currentA1*pos2.x,
								b1*pos2.y - b2*(angle1v.z + angle2v.z),
								b1*pos2.z + b2*(angle1v.y + angle2v.y)); //Use Curstress instead of -a1*Pos2.x to account for non-linear deformation 
	forcePos = -forceNeg;

	momentNeg = TI_Vec3D<double> (	a2*(angle2v.x - angle1v.x),
								-b2*pos2.z - b3*(2*angle1v.y + angle2v.y),
								b2*pos2.y - b3*(2*angle1v.z + angle2v.z));
	momentPos = TI_Vec3D<double> (	a2*(angle1v.x - angle2v.x),
								-b2*pos2.z - b3*(angle1v.y + 2*angle2v.y),
								b2*pos2.y - b3*(angle1v.z + 2*angle2v.z));
	//local damping:
	if (isLocalVelocityValid()){ //if we don't have the basis for a good damping calculation, don't do any damping.
				
		float sqA1=mat->_sqA1, sqA2xIp=mat->_sqA2xIp,sqB1=mat->_sqB1, sqB2xFMp=mat->_sqB2xFMp, sqB3xIp=mat->_sqB3xIp;
		TI_Vec3D<double> posCalc(	sqA1*dPos2.x,
								sqB1*dPos2.y - sqB2xFMp*(dAngle1.z+dAngle2.z),
								sqB1*dPos2.z + sqB2xFMp*(dAngle1.y+dAngle2.y));
		
		forceNeg += pVNeg->dampingMultiplier()*posCalc;
		forcePos -= pVPos->dampingMultiplier()*posCalc;
		

		momentNeg -= 0.5*pVNeg->dampingMultiplier()*TI_Vec3D<>(	-sqA2xIp*(dAngle2.x - dAngle1.x),
																sqB2xFMp*dPos2.z + sqB3xIp*(2*dAngle1.y + dAngle2.y),
																-sqB2xFMp*dPos2.y + sqB3xIp*(2*dAngle1.z + dAngle2.z));		
		momentPos -= 0.5*pVPos->dampingMultiplier()*TI_Vec3D<>(	sqA2xIp*(dAngle2.x - dAngle1.x),
																sqB2xFMp*dPos2.z + sqB3xIp*(dAngle1.y + 2*dAngle2.y),
																-sqB2xFMp*dPos2.y + sqB3xIp*(dAngle1.z + 2*dAngle2.z));

	}
	else setBoolState(LOCAL_VELOCITY_VALID, true); //we're good for next go-around unless something changes
	//	transform forces and moments to local voxel coordinates
	if (!smallAngle){
		forceNeg = angle1.RotateVec3DInv(forceNeg);
		momentNeg = angle1.RotateVec3DInv(momentNeg);
	}
	forcePos = angle2.RotateVec3DInv(forcePos);
	momentPos = angle2.RotateVec3DInv(momentPos);

	toAxisOriginal(&forceNeg);
	toAxisOriginal(&forcePos);
	toAxisOriginal(&momentNeg);
	toAxisOriginal(&momentPos);
	//time 214.72us

	// assert(!(forceNeg.x != forceNeg.x) || !(forceNeg.y != forceNeg.y) || !(forceNeg.z != forceNeg.z)); //assert non QNAN
	// assert(!(forcePos.x != forcePos.x) || !(forcePos.y != forcePos.y) || !(forcePos.z != forcePos.z)); //assert non QNAN
}


CUDA_DEVICE float TI_Link::updateStrain(float axialStrain)
{
	int di = 0;
	
	strain = axialStrain; //redundant?
	

	if (mat->linear){
		
		if (axialStrain > maxStrain) maxStrain = axialStrain; //remember this maximum for easy reference
		
		return mat->stress(axialStrain, currentTransverseStrainSum);
	}
	else {
		float returnStress;
		

		if (axialStrain > maxStrain){ //if new territory on the stress/strain curve
			maxStrain = axialStrain; //remember this maximum for easy reference
			returnStress = mat->stress(axialStrain, currentTransverseStrainSum);
			
			
			if (mat->nu != 0.0f) strainOffset = maxStrain-mat->stress(axialStrain)/(mat->_eHat*(1-mat->nu)); //precalculate strain offset for when we back off
			else strainOffset = maxStrain-returnStress/mat->E; //precalculate strain offset for when we back off

		}
		else { //backed off a non-linear material, therefore in linear region.
			
			float relativeStrain = axialStrain-strainOffset; // treat the material as linear with a strain offset according to the maximum plastic deformation
			
			if (mat->nu != 0.0f) returnStress = mat->stress(relativeStrain, currentTransverseStrainSum, true);
			else returnStress = mat->E*relativeStrain;
		}
		
		return returnStress;

	}

}

CUDA_DEVICE float TI_Link::strainEnergy() const
{
	return	forceNeg.x*forceNeg.x/(2.0f*mat->_a1) + //Tensile strain
			momentNeg.x*momentNeg.x/(2.0*mat->_a2) + //Torsion strain
			(momentNeg.z*momentNeg.z - momentNeg.z*momentPos.z +momentPos.z*momentPos.z)/(3.0*mat->_b3) + //Bending Z
			(momentNeg.y*momentNeg.y - momentNeg.y*momentPos.y +momentPos.y*momentPos.y)/(3.0*mat->_b3); //Bending Y
}

CUDA_DEVICE float TI_Link::axialStiffness() {
	if (mat->isXyzIndependent()) return mat->_a1;
	else {
		updateRestLength();
		updateTransverseInfo();

		return (float)(mat->_eHat*currentTransverseArea/((strain+1)*currentRestLength)); // _a1;
	}
} 

CUDA_DEVICE float TI_Link::a1() const {return mat->_a1;}
CUDA_DEVICE float TI_Link::a2() const {return mat->_a2;}
CUDA_DEVICE float TI_Link::b1() const {return mat->_b1;}
CUDA_DEVICE float TI_Link::b2() const {return mat->_b2;}
CUDA_DEVICE float TI_Link::b3() const {return mat->_b3;}

