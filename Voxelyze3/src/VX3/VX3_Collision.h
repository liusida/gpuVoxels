#if !defined(VX3_COLLISION_H)
#define VX3_COLLISION_H
#include "VX3.cuh"
#include "VX3_Voxel.h"

class VX3_Collision
{
public:
	__device__ VX3_Collision(VX3_Voxel* v1, VX3_Voxel* v2); //!< Constructor taking the two voxels to watch for collision between. The order is irrelevant. @param[in] v1 One voxel @param[in] v2 The other voxel
	// VX3_Collision& operator=(const VX3_Collision& col); //!< Overload "=" operator.
	// VX3_Collision(const VX3_Collision& col) {*this = col;} //!< copy constructor.

	__device__ VX3_Vec3D<double> const contactForce(VX3_Voxel* pVoxel); //!< Returns the repelling force acting on pVoxel from the penetration of the other voxel if their collision boundaries overlap. Otherwise returns a zero force. This force will only be accurate if updateContactForce() has been called since the voxels last moved. @param[in] pVoxel The voxel in question. This should be voxel1() or voxel2() to have any meaning. Otherwise a zero force is returned.
	__device__ void updateContactForce(); //!< Updates the state this collision based on the current positions and properties of voxel1() and voxel2(). Call contactForce() with either voxel as the argument to obtain the repelling penetration force (if any) that exisits.

	double envelopeRadius = 0.625f; //!<The collision envelope radius that these two voxels collide at. Even though voxels are cubic, a spherical collision envelope is used for computation efficiency. Values are multiplied by the length of an edge of the voxel to determine the actual collision radius. Values less than 0.5 or greater than 0.866 are probably of limited use. Prefer around 0.625 (default).

	VX3_Voxel *pV1, *pV2;
	double penetrationStiff; //in N/m for these two voxels
	double dampingC; //damping factor for these two voxels
	VX3_Vec3D<double> force;
};

#endif //VX3_COLLISION_H