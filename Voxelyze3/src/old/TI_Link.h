#if !defined(TI_LINK_H)
#define TI_LINK_H
#include "TI_Utils.h"

#include "VX_Link.h"

class VX3_VoxelyzeKernel;
class TI_VoxelyzeKernel;
class TI_Voxel;
class TI_MaterialLink;

class TI_Link
{
public:
    TI_Link(CVX_Link* p, VX3_VoxelyzeKernel* k);

    TI_Voxel* getDevPtrFromHostPtr(CVX_Voxel* p);

	CUDA_DEVICE void reset(); //!< Resets all current state information about this link to the initial value.

	CUDA_DEVICE TI_Voxel* voxel(bool positiveEnd) const {return positiveEnd?pVPos:pVNeg;} //!< Returns a pointer to one of the two voxels that compose this link. @param[in] positiveEnd Specifies which voxel is desired.
	CUDA_DEVICE TI_Vec3D<> force(bool positiveEnd) const {return positiveEnd?forcePos:forceNeg;} //!< Returns the current force acting on a voxel due to the position and orientation of the other. @param[in] positiveEnd Specifies which voxel information is desired about.
	CUDA_DEVICE TI_Vec3D<> moment(bool positiveEnd) const {return positiveEnd?momentPos:momentNeg;} //!< Returns the current moment acting on a voxel due to the position and orientation of the other. @param[in] positiveEnd Specifies which voxel information is desired about.


	CUDA_DEVICE float axialStrain() const {return strain;} //!< returns the current overall axial strain (unitless) between the two voxels.
	CUDA_DEVICE float axialStrain(bool positiveEnd) const; //!< Returns the current calculated axial strain of the half of the link contained in the specified voxel. @param[in] positiveEnd Specifies which voxel information is desired about.
	CUDA_DEVICE float axialStress() const {return _stress;} //!< returns the current overall true axial stress (MPa) between the two voxels.

	CUDA_DEVICE bool isSmallAngle() const {return smallAngle;} //!< Returns true if this link is currently operating with a small angle assumption.
	CUDA_DEVICE bool isYielded() const; //!< Returns true if the stress on this bond has ever exceeded its yield stress
	CUDA_DEVICE bool isFailed() const; //!< Returns true if the stress on this bond has ever exceeded its failure stress

	CUDA_DEVICE float strainEnergy() const; //!< Calculates and return the strain energy of this link according to current forces and moments. (units: Joules, or Kg m^2 / s^2)
	CUDA_DEVICE float axialStiffness(); //!< Calculates and returns the current linear axial stiffness of this link at it's current strain.

	CUDA_DEVICE void updateForces(); //!< Called every timestep to calculate the forces and moments acting between the two constituent voxels in their current relative positions and orientations.

	CUDA_DEVICE void updateRestLength(); //!< Updates the rest length of this voxel. Call this every timestep where the nominal size of either voxel may have changed, due to actuation or thermal expansion.
	CUDA_DEVICE void updateTransverseInfo(); //!< Updates information about this voxel pertaining to volumetric deformations. Call this every timestep if the poisson's ratio of the link material is non-zero.

	CUDA_DEVICE float updateStrain(float axialStrain); //updates strainNeg and strainPos according to the provided axial strain. returns current stress as well (MPa)

	CUDA_DEVICE bool isLocalVelocityValid() const {return boolStates & LOCAL_VELOCITY_VALID ? true : false;} //
	CUDA_DEVICE void setBoolState(linkFlags flag, bool set=true) {set ? boolStates |= (int)flag : boolStates &= ~(int)flag;}

	//beam parameters
	CUDA_DEVICE float a1() const;
	CUDA_DEVICE float a2() const;
	CUDA_DEVICE float b1() const;
	CUDA_DEVICE float b2() const;
	CUDA_DEVICE float b3() const;

	CUDA_DEVICE TI_Quat3D<double> orientLink(/*double restLength*/); //updates pos2, angle1, angle2, and smallAngle. returns the rotation quaternion (after toAxisX) used to get to this orientation

	//unwind a coordinate as if the bond was in the the positive X direction (and back...)
	template <typename T> CUDA_DEVICE void toAxisX			(TI_Vec3D<T>* const pV) const {switch (axis){case Y_AXIS: {T tmp = pV->x; pV->x=pV->y; pV->y = -tmp; break;} case Z_AXIS: {T tmp = pV->x; pV->x=pV->z; pV->z = -tmp; break;} default: break;}} //transforms a TI_vec3D in the original orientation of the bond to that as if the bond was in +X direction
	template <typename T> CUDA_DEVICE void toAxisX			(TI_Quat3D<T>* const pQ) const {switch (axis){case Y_AXIS: {T tmp = pQ->x; pQ->x=pQ->y; pQ->y = -tmp; break;} case Z_AXIS: {T tmp = pQ->x; pQ->x=pQ->z; pQ->z = -tmp; break;} default: break;}}
	template <typename T> CUDA_DEVICE TI_Vec3D<T> toAxisX		(const TI_Vec3D<T>& v)	const {switch (axis){case Y_AXIS: return TI_Vec3D<T>(v.y, -v.x, v.z); case Z_AXIS: return TI_Vec3D<T>(v.z, v.y, -v.x); default: return v;}} //transforms a TI_vec3D in the original orientation of the bond to that as if the bond was in +X direction
	template <typename T> CUDA_DEVICE TI_Quat3D<T> toAxisX		(const TI_Quat3D<T>& q)	const {switch (axis){case Y_AXIS: return TI_Quat3D<T>(q.w, q.y, -q.x, q.z); case Z_AXIS: return TI_Quat3D<T>(q.w, q.z, q.y, -q.x); default: return q;}} //transforms a TI_vec3D in the original orientation of the bond to that as if the bond was in +X direction
	template <typename T> CUDA_DEVICE void toAxisOriginal	(TI_Vec3D<T>* const pV) const {switch (axis){case Y_AXIS: {T tmp = pV->y; pV->y=pV->x; pV->x = -tmp; break;} case Z_AXIS: {T tmp = pV->z; pV->z=pV->x; pV->x = -tmp; break;} default: break;}}
	template <typename T> CUDA_DEVICE void toAxisOriginal	(TI_Quat3D<T>* const pQ) const {switch (axis){case Y_AXIS: {T tmp = pQ->y; pQ->y=pQ->x; pQ->x = -tmp; break;} case Z_AXIS: {T tmp = pQ->z; pQ->z=pQ->x; pQ->x = -tmp; break;} default: break;}}

    CUDA_DEVICE void test();



    // CUDA_DEVICE void updateForces();
	// CUDA_DEVICE TI_Quat3D<double> orientLink(/*double restLength*/); //updates pos2, angle1, angle2, and smallAngle. returns the rotation quaternion (after toAxisX) used to get to this orientation

	// template <typename T>
    // CUDA_DEVICE TI_Vec3D<T> toAxisX		(const TI_Vec3D<T>& v)	const {switch (axis){case Y_AXIS: return TI_Vec3D<T>(v.y, -v.x, v.z); case Z_AXIS: return TI_Vec3D<T>(v.z, v.y, -v.x); default: return v;}} //transforms a TI_vec3D in the original orientation of the bond to that as if the bond was in +X direction

/*data*/

    CVX_Link* _link;
    VX3_VoxelyzeKernel* _kernel;

    TI_Voxel *pVNeg, *pVPos;
	TI_Vec3D<> forceNeg, forcePos;
	TI_Vec3D<> momentNeg, momentPos;

	float strain;
	float maxStrain, /*maxStrainRatio,*/ strainOffset; //keep track of the maximums for yield/fail/nonlinear materials (and the ratio of the maximum from 0 to 1 [all positive end strain to all negative end strain])

	linkState boolStates;			//single int to store many boolean state values as bit flags according to 
	
	linkAxis axis;

	TI_MaterialLink* mat;
	float strainRatio; //ration of Epos to Eneg (EPos/Eneg)

	TI_Vec3D<double> pos2, angle1v, angle2v; //pos1 is always = 0,0,0
	TI_Quat3D<double> angle1, angle2; //this bond in local coordinates. 
	bool smallAngle; //based on compiled precision setting
	double currentRestLength;
	float currentTransverseArea, currentTransverseStrainSum; //so we don't have to re-calculate everytime

	float _stress; //keep this around for convenience

};



#endif // TI_LINK_H
