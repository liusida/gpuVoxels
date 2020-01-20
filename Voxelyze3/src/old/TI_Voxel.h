#if !defined(TI_VOXEL_H)
#define TI_VOXEL_H
#include "TI_Utils.h"

#include "VX_Voxel.h"

#include "TI_External.h"
#include "TI_MaterialVoxel.h"
#include "TI_Link.h"
#include "TI_Collision.h"

class TI_Collision;
class VX3_VoxelyzeKernel;

class TI_Voxel {
public:
	TI_Voxel()=default;
    TI_Voxel(CVX_Voxel* p, VX3_VoxelyzeKernel* k);
	~TI_Voxel();
    TI_Link* getDevPtrFromHostPtr(CVX_Link* p);

	CUDA_DEVICE TI_Link* link(linkDirection direction) const {return links[direction];} //!< Returns a pointer to the link object in the specified direction if it exists. Returns null if a link does not exist in this direction.
	CUDA_DEVICE int linkCount() const {int retVal =0; for (int i=0; i<6; i++) if (links[i]) retVal++; return retVal;} //!< Returns the number of links present for this voxel out of a total 6 possible.
	CUDA_DEVICE TI_Voxel* adjacentVoxel(linkDirection direction) const; //!< Returns a pointer to the voxel in the specified direction if one exists, or NULL otherwise. @param[in] direction Positive or negative X, Y, or Z direction according to the linkDirection enum.
	CUDA_DEVICE short indexX() {return ix;} //!< Returns the global X index of this voxel.
	CUDA_DEVICE short indexY() {return iy;} //!< Returns the global Y index of this voxel.
	CUDA_DEVICE short indexZ() {return iz;} //!< Returns the global Z index of this voxel.

	CUDA_DEVICE TI_MaterialVoxel* material() {return mat;} //!<Returns the linked material object containing the physical properties of this voxel.
	
	CUDA_DEVICE bool externalExists() {return ext?true:false;} //!< Returns true if this voxel has had its TI_External object created. This does not mecessarily imply that this external object actually contains any fixes or forces.
	CUDA_DEVICE TI_External* external() {if (!ext) ext = new TI_External(); return ext;} //!< Returns a pointer to this voxel's unique external object that contains fixes, forces, and/or displacements. Allocates a new empty one if it doesn't already exist. Use externalExists() to determine if external() has been previously called at any time.

	CUDA_DEVICE void timeStep(float dt); //!< Advances this voxel's state according to all forces and moments acting on it. Large timesteps will cause instability. Use CVoxelyze::recommendedTimeStep() to get the recommended largest stable timestep. @param[in] dt Timestep (in second) to advance.

	//physical location
	CUDA_DEVICE TI_Vec3D<double> position() const {return pos;} //!< Returns the center position of this voxel in meters (GCS). This is the origin of the local coordinate system (LCS).
	CUDA_DEVICE TI_Vec3D<double> originalPosition() const {double s=mat->nominalSize(); return TI_Vec3D<double>(ix*s, iy*s, iz*s);} //!< Returns the initial (nominal) position of this voxel.
	CUDA_DEVICE TI_Vec3D<double> displacement() const {return (pos - originalPosition());} //!< Returns the 3D displacement of this voxel from its original location in meters (GCS)/
	CUDA_DEVICE TI_Vec3D<float> size() const {return cornerOffset(PPP)-cornerOffset(NNN);} //!< Returns the current deformed size of this voxel in the local voxel coordinates system (LCS). If asymmetric forces are acting on this voxel, the voxel may not be centered on position(). Use cornerNegative() and cornerPositive() to determine this information.
	CUDA_DEVICE TI_Vec3D<float> cornerPosition(voxelCorner corner) const; //!< Returns the deformed location of the voxel corner in the specified corner in the global coordinate system (GCS). Essentially cornerOffset() with the voxel's current global position/rotation applied.
	CUDA_DEVICE TI_Vec3D<float> cornerOffset(voxelCorner corner) const; //!< Returns the deformed location of the voxel corner in the specified corner in the local voxel coordinate system (LCS). Used to draw the deformed voxel in the correct position relative to the position().
	CUDA_DEVICE bool isInterior() const {return (boolStates & SURFACE)?true:false;} //!< Returns true if the voxel is surrounded by other voxels on its 6 coordinate faces. Returns false if 1 or more faces are exposed.
	CUDA_DEVICE bool isSurface() const {return !isInterior();} //!< Convenience function to enhance code readibility. The inverse of isInterior(). Returns true 1 or more faces are exposed. Returns false if the voxel is surrounded by other voxels on its 6 coordinate faces.

	CUDA_DEVICE TI_Vec3D<double> baseSize() const {return mat->size()*(1+tempe*mat->alphaCTE);} //!<Returns the nominal size of this voxel (LCS) accounting for any specified temperature and external actuation. Specifically, returns the zero-stress size of the voxel if all forces/moments were removed.
	CUDA_DEVICE double baseSize(linkAxis axis) const {return mat->size()[axis]*(1+tempe*mat->alphaCTE);} //!<Returns the nominal size of this voxel in the specified axis accounting for any specified temperature and external actuation. Specifically, returns the zero-stress dimension of the voxel if all forces/moments were removed.
	CUDA_DEVICE double baseSizeAverage() const {TI_Vec3D<double> bSize=baseSize(); return (bSize.x+bSize.y+bSize.z)/3.0f;} //!<Returns the average nominal size of the voxel in a zero-stress (no force) state. (X+Y+Z/3)

	CUDA_DEVICE TI_Quat3D<double> orientation() const {return orient;} //!< Returns the orientation of this voxel in quaternion form (GCS). This orientation defines the relative orientation of the local coordinate system (LCS). The unit quaternion represents the original orientation of this voxel.
	CUDA_DEVICE float orientationAngle() const {return (float)orient.Angle();} //!< Use with orientationAxis() to get the orientation of this voxel in angle/axis form. Returns the angle in radians.
	CUDA_DEVICE TI_Vec3D<double> orientationAxis() const {return orient.Axis();} //!< Use with orientationAngle() to get the orientation of this voxel in angle/axis form. Returns a unit vector in the global coordinate system (GCS).

	CUDA_DEVICE float displacementMagnitude() const {return (float)displacement().Length();} //!< Returns the distance (magnitude of displacement) this voxel has moved from its initial nominal position. (GCS)
	CUDA_DEVICE float angularDisplacementMagnitude() const {return (float)orient.Angle();} //!< Returns the angle (magnitude of angular displacement) this voxel has rotated from its initial nominal orientation. (GCS)
	CUDA_DEVICE TI_Vec3D<double> velocity() const {return linMom*mat->_massInverse;} //!< Returns the 3D velocity of this voxel in m/s (GCS)
	CUDA_DEVICE float velocityMagnitude() const {return (float)(linMom.Length()*mat->_massInverse);} //!< Returns the velocity of this voxel in m/s.
	CUDA_DEVICE TI_Vec3D<double> angularVelocity() const {return angMom*mat->_momentInertiaInverse;} //!< Returns the 3D angular velocity of this voxel in rad/s (GCS)
	CUDA_DEVICE float angularVelocityMagnitude() const {return (float)(angMom.Length()*mat->_momentInertiaInverse);} //!< Returns the angular velocity of this voxel in rad/s.
	CUDA_DEVICE float kineticEnergy() const {return (float)(0.5*(mat->_massInverse*linMom.Length2() + mat->_momentInertiaInverse*angMom.Length2()));} //!< Returms the kinetic energy of this voxel in Joules.
	CUDA_DEVICE float volumetricStrain() const {return (float)(strain(false).x+strain(false).y+strain(false).z);} //!< Returns the volumetric strain of the voxel according to the definition at http://www.colorado.edu/engineering/CAS/courses.d/Structures.d/IAST.Lect05.d/IAST.Lect05.pdf
	CUDA_DEVICE float pressure() const {return -mat->youngsModulus()*volumetricStrain()/(3*(1-2*mat->poissonsRatio()));} //!< Returns the engineering internal "pressure" in Pa according to the definition at http://www.colorado.edu/engineering/CAS/courses.d/Structures.d/IAST.Lect05.d/IAST.Lect05.pdf

	//material state
	CUDA_DEVICE bool isYielded() const; //!< Returns true if the stress in this voxel has ever exceeded the yield stress. Technically, this returns true if any of the connected links have yielded since the stress state of the voxel is never expressly calculated.
	CUDA_DEVICE bool isFailed() const; //!< Returns true if the stress in this voxel has ever exceeded the failure stress. Technically, this returns true if any of the connected links have failed since the stress state of the voxel is never expressly calculated.

	//@ voxel level for heat diffusion experiments later
	CUDA_DEVICE float temperature() {return tempe;} //!< Returns the current temperature of this voxel in degrees Celsius.
	CUDA_DEVICE void setTemperature(float temperature); //!< Specifies the temperature for this voxel. This adds (or subtracts) the correct amount of thermal energy to leave the voxel at ths specified temperature, but this temperature will not be maintaned without subsequent determines the amount of scaling from the temperature

	CUDA_DEVICE TI_Vec3D<float> externalForce(); //!< Returns the current external force applied to this voxel in newtons. If the voxel is not fixed this will return any applied external forces. If fixed it will return the current reaction force necessary to enforce the zero-motion constraint.
	CUDA_DEVICE TI_Vec3D<float> externalMoment(); //!< Returns the current external moment applied to this voxel in N-m. If the voxel is not fixed this will return any applied external moments. If fixed it will return the current reaction moment necessary to enforce the zero-motion constraint.

	CUDA_DEVICE void haltMotion(){linMom = angMom = TI_Vec3D<>(0,0,0);} //!< Halts all momentum of this block. Unless fixed the voxel will continue to move in subsequent timesteps.

	CUDA_DEVICE void enableFloor(bool enabled) {enabled ? boolStates |= FLOOR_ENABLED : boolStates &= ~FLOOR_ENABLED;} //!< Enables this voxel interacting with the floor at Z=0. @param[in] enabled Enable interaction
	CUDA_CALLABLE_MEMBER bool isFloorEnabled() const {return boolStates & FLOOR_ENABLED ? true : false;} //!< Returns true of this voxel will interact with the floor at Z=0.
	CUDA_DEVICE bool isFloorStaticFriction() const {return boolStates & FLOOR_STATIC_FRICTION ? true : false;} //!< Returns true if this voxel is in contact with the floor and stationary in the horizontal directions. This corresponds to that voxel being in the mode of static friction (as opposed to kinetic) with the floor.
	CUDA_DEVICE float floorPenetration() const {return (float)(baseSizeAverage()/2 - mat->nominalSize()/2 - pos.z);} //!< Returns the interference (in meters) between the collision envelope of this voxel and the floor at Z=0. Positive numbers correspond to interference. If the voxel is not touching the floor 0 is returned.

	CUDA_DEVICE TI_Vec3D<double> force(); //!< Calculates and returns the sum of the current forces on this voxel. This would normally only be called internally, but can be used to query the state of a voxel for visualization or debugging.
	CUDA_DEVICE TI_Vec3D<double> moment(); //!< Calculates and returns the sum of the current moments on this voxel. This would normally only be called internally, but can be used to query the state of a voxel for visualization or debugging.

	CUDA_DEVICE float transverseArea(linkAxis axis); //!< Returns the transverse area of this voxel with respect to the specified axis. This would normally be called only internally, but can be used to calculate the correct relationship between force and stress for this voxel if Poisson's ratio is non-zero.
	CUDA_DEVICE float transverseStrainSum(linkAxis axis); //!< Returns the sum of the current strain of this voxel in the two mutually perpindicular axes to the specified axis. This would normally be called only internally, but can be used to correctly calculate stress for this voxel if Poisson's ratio is non-zero.

	CUDA_DEVICE float dampingMultiplier() {return 2*mat->_sqrtMass*mat->zetaInternal/previousDt;} //!< Returns the damping multiplier for this voxel. This would normally be called only internally for the internal damping calculations.

	//a couple global convenience functions to have wherever the link enums are used
	CUDA_DEVICE static inline linkAxis toAxis(linkDirection direction) {return (linkAxis)((int)direction/2);} //!< Returns the link axis of the specified link direction.
	CUDA_DEVICE static inline linkDirection toDirection(linkAxis axis, bool positiveDirection) {return (linkDirection)(2*((int)axis) + positiveDirection?0:1);} //!< Returns the link direction of the specified link axis and sign.
	CUDA_DEVICE static inline bool isNegative(linkDirection direction) {return direction%2==1;} //!< Returns true if the specified link direction is negative.
	CUDA_DEVICE static inline bool isPositive(linkDirection direction) {return direction%2==0;} //!< Returns true if the specified link direction is positive.
	CUDA_DEVICE static inline linkDirection toOpposite(linkDirection direction) {return (linkDirection)(direction-direction%2 + (direction+1)%2);} //!< Returns the opposite (negated) link direction of the specified direction.

	CUDA_DEVICE void replaceMaterial(TI_MaterialVoxel* newMaterial); //!<Replaces the material properties of this voxel (but not links) to this new CVX_Material. May cause unexpected behavior if certain material properties are changed mid-simulation. @param [in] newMaterial The new material properties for this voxel.

	CUDA_DEVICE void addLinkInfo(linkDirection direction, TI_Link* link); //adds the information about a link connected to this voxel in the specified direction
	CUDA_DEVICE void removeLinkInfo(linkDirection direction); //removes the information about a link connected to this voxel in the specified direction

	CUDA_DEVICE void setFloorStaticFriction(bool active) {active? boolStates |= FLOOR_STATIC_FRICTION : boolStates &= ~FLOOR_STATIC_FRICTION;}

	CUDA_DEVICE void floorForce(float dt, TI_Vec3D<double>* pTotalForce); //modifies pTotalForce to include the object's interaction with a floor. This should be calculated as the last step of sumForce so that pTotalForce is complete.

	CUDA_DEVICE TI_Vec3D<float> strain(bool poissonsStrain) const; //LCS returns voxel strain. if tensionStrain true and no actual tension in that
	CUDA_DEVICE TI_Vec3D<float> poissonsStrain();

	CUDA_DEVICE void eulerStep(float dt); //execute an euler time step at the specified dt

	CUDA_DEVICE void updateSurface();
	CUDA_DEVICE void enableCollisions(bool enabled, float watchRadius = 0.0f); //watchRadius in voxel units
	CUDA_DEVICE bool isCollisionsEnabled() const {return boolStates & COLLISIONS_ENABLED ? true : false;}
	CUDA_DEVICE void generateNearby(int linkDepth, int gindex, bool surfaceOnly = true);

/* data */
    CVX_Voxel* _voxel;
    VX3_VoxelyzeKernel* _kernel;

	TI_MaterialVoxel* mat = NULL;
	short ix, iy, iz;
	TI_External* ext = NULL;

	TI_Link* links[6]; //links in the 6 cardinal directions according to linkDirection enumeration

	//voxel state
	TI_Vec3D<double> pos;					//current center position (meters) (GCS)
	TI_Vec3D<double> linMom;				//current linear momentum (kg*m/s) (GCS)
	TI_Quat3D<double> orient;				//current orientation (GCS)
	TI_Vec3D<double> angMom;				//current angular momentum (kg*m^2/s) (GCS)

	voxState boolStates;				//single int to store many boolean state values as bit flags according to 

	float tempe; //0 is no expansion

	
	TI_Vec3D<float> pStrain; //cached poissons strain
	bool poissonsStrainInvalid; //flag for recomputing poissons strain.

	float previousDt; //remember the duration of the last timestep of this voxel

	TI_Vec3D<float> lastColWatchPosition;
	// TI_vector<TI_Collision*> colWatch;
	// TI_vector<TI_Voxel*> nearby;

};

#endif // TI_VOXEL_H
