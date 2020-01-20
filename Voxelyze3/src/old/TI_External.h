#if !defined(TI_EXTERNAL_H)
#define TI_EXTERNAL_H

#include "TI_Utils.h"
#include "VX_External.h"

class TI_External {
public:
	TI_External( CVX_External* p );

	CUDA_DEVICE inline void dofSet(dofObject& obj, dofComponent dof, bool set) {set ? obj|=dof : obj&=~dof;}
	CUDA_DEVICE inline void dofSetAll(dofObject& obj, bool set) {set ? obj|=0x3F : obj&=~0x3F;}
	CUDA_DEVICE inline bool dofIsSet(dofObject obj, dofComponent dof){return (dof&obj)?true:false;}
	CUDA_DEVICE inline bool dofIsAllSet(dofObject obj){return (obj&0x3F)==0x3F;}
	CUDA_DEVICE inline bool dofIsNoneSet(dofObject obj){return !(obj&0x3F);}
	CUDA_DEVICE inline dofObject dof(bool tx, bool ty, bool tz, bool rx, bool ry, bool rz) {dofObject ret=0; dofSet(ret, X_TRANSLATE, tx); dofSet(ret, Y_TRANSLATE, ty); dofSet(ret, Z_TRANSLATE, tz); dofSet(ret, X_ROTATE, rx); dofSet(ret, Y_ROTATE, ry); dofSet(ret, Z_ROTATE, rz); return ret;}

	CUDA_DEVICE TI_External();
	CUDA_DEVICE TI_External(const TI_External& eIn) {*this = eIn;} //!< Copy constructor
	CUDA_DEVICE TI_External& operator=(const TI_External& eIn); //!< Equals operator
	CUDA_DEVICE inline bool operator==(const TI_External& b) {return dofFixed==b.dofFixed && extForce==b.extForce && extMoment==b.extMoment && extTranslation==b.extTranslation && extRotation==b.extRotation;} //!< comparison operator

	CUDA_DEVICE void reset(); //!< Resets this external to defaults - i.e., no effect on a voxel  (forces, fixed, displacements, etc) 
	CUDA_DEVICE bool isEmpty() {return (dofFixed == 0 && extForce==TI_Vec3D<float>() && extMoment==TI_Vec3D<float>());} //!< returns true if this external is empty - i.e is exerting no effect on a voxel

	CUDA_DEVICE bool isFixed(dofComponent dof) {return dofIsSet(dofFixed, dof);}  //!< Returns true if the specified degree of freedom is fixed for this voxel. @param[in] dof Degree of freedom to query according to the dofComponent enum.
	CUDA_DEVICE bool isFixedAll() {return dofIsAllSet(dofFixed);} //!< Returns true if all 6 degrees of freedom are fixed for this voxel.
	CUDA_DEVICE bool isFixedAllTranslation() {return dofIsSet(dofFixed, X_TRANSLATE) && dofIsSet(dofFixed, Y_TRANSLATE) && dofIsSet(dofFixed, Z_TRANSLATE);} //!< Returns true if all translational degrees of freedom are fixed.
	CUDA_DEVICE bool isFixedAllRotation() {return dofIsSet(dofFixed, X_ROTATE) && dofIsSet(dofFixed, Y_ROTATE) && dofIsSet(dofFixed, Z_ROTATE);} //!< Returns true if all rotationsl degrees of freedom are fixed.
	
	CUDA_DEVICE bool isFixedAny() {return (dofFixed != 0);} //!< Returns true if any of the 6 degrees of freedom are fixed for this voxel.
	CUDA_DEVICE bool isFixedAnyTranslation() {return dofIsSet(dofFixed, X_TRANSLATE) || dofIsSet(dofFixed, Y_TRANSLATE) || dofIsSet(dofFixed, Z_TRANSLATE);} //!< Returns true if any of the three translational degrees of freedom are fixed.
	CUDA_DEVICE bool isFixedAnyRotation() {return dofIsSet(dofFixed, X_ROTATE) || dofIsSet(dofFixed, Y_ROTATE) || dofIsSet(dofFixed, Z_ROTATE);} //!< Returns true if any of the three rotational degrees of freedom are fixed.

	CUDA_DEVICE TI_Vec3D<double> translation() {return extTranslation;} //!< Returns any external translation that has been applied to this external.
	CUDA_DEVICE TI_Vec3D<double> rotation() {return extRotation;} //!< Returns any external rotation that has been applied to this external as a rotation vector.
	CUDA_DEVICE TI_Quat3D<double> rotationQuat() {return _extRotationQ;} //!< Returns any external rotation that has been applied to this external as a quaternion.


	CUDA_DEVICE void setFixed(bool xTranslate, bool yTranslate, bool zTranslate, bool xRotate, bool yRotate, bool zRotate); //!< Sets any of the degrees of freedom specified as "true" to fixed for this voxel. (GCS) @param[in] xTranslate Translation in the X direction  @param[in] yTranslate Translation in the Y direction @param[in] zTranslate Translation in the Z direction @param[in] xRotate Rotation about the X axis @param[in] yRotate Rotation about the Y axis @param[in] zRotate Rotation about the Z axis
	CUDA_DEVICE void setFixed(dofComponent dof, bool fixed=true) {fixed?setDisplacement(dof):clearDisplacement(dof);} //!< Sets the specified degree of freedom to either fixed or free, depending on the value of fixed. Either way, sets the translational or rotational displacement of this degree of freedom to zero. @param[in] dof the degree of freedom in question @param[in] fixed Whether this degree of freedom should be fixed (true) or free (false).
	CUDA_DEVICE void setFixedAll(bool fixed=true) {fixed?setDisplacementAll():clearDisplacementAll();} //!< Sets all 6 degrees of freedom to either fixed or free depending on the value of fixed. Either way, sets all displacements to zero. @param[in] fixed Whether all degrees of freedom should be fixed (true) or free (false).

	CUDA_DEVICE void setDisplacement(dofComponent dof, double displacement=0.0); //!< Fixes the specified degree of freedom and applies the prescribed displacement if specified. @param[in] dof the degree of freedom in question. @param[in] displacement The displacement in meters (translational dofs) or radians (rotational dofs) to apply. Large fixed displacements may cause instability.
	CUDA_DEVICE void setDisplacementAll(const TI_Vec3D<double>& translation = TI_Vec3D<double>(0,0,0), const TI_Vec3D<double>& rotation = TI_Vec3D<double>(0,0,0)); //!< Fixes the all degrees of freedom and applies the specified translation and rotation. @param[in] translation The translation in meters from this voxel's nominal position to fix it at. @param[in] rotation The rotation (in the form of a rotation vector) from this voxel's nominal orientation to fix it at.

	CUDA_DEVICE void clearDisplacement(dofComponent dof); //!< Clears any prescribed displacement from this degree of freedom and unfixes it, too. @param[in] dof the degree of freedom in question.
	CUDA_DEVICE void clearDisplacementAll(); //!< Clears all prescribed displacement from this voxel and completely unfixes it, too.

	CUDA_DEVICE TI_Vec3D<float> force() {return extForce;} //!< Returns the current applied external force in newtons.
	CUDA_DEVICE TI_Vec3D<float> moment() {return extMoment;} //!< Returns the current applied external moment in N-m.

	CUDA_DEVICE void setForce(const float xForce, const float yForce, const float zForce) {extForce = TI_Vec3D<float>(xForce, yForce, zForce);} //!< Applies forces to this voxel in the global coordinate system. Has no effect in any fixed degrees of freedom. @param xForce Force in the X direction in newtons.  @param yForce Force in the Y direction in newtons.  @param zForce Force in the Z direction in newtons. 
	CUDA_DEVICE void setForce(const TI_Vec3D<float>& force) {extForce = force;} //!< Convenience function for setExternalForce(float, float, float).
	CUDA_DEVICE void setMoment(const float xMoment, const float yMoment, const float zMoment) {extMoment = TI_Vec3D<float>(xMoment, yMoment, zMoment);}  //!< Applies moments to this voxel in the global coordinate system. All rotations according to the right-hand rule. Has no effect in any fixed degrees of freedom. @param xMoment Moment in the X axis rotation in newton-meters. @param yMoment Moment in the Y axis rotation in newton-meters. @param zMoment Moment in the Z axis rotation in newton-meters. 
	CUDA_DEVICE void setMoment(const TI_Vec3D<float>& moment) {extMoment = moment;} //!< Convenience function for setExternalMoment(float, float, float).

	CUDA_DEVICE void addForce(const float xForce, const float yForce, const float zForce) {extForce += TI_Vec3D<float>(xForce, yForce, zForce);} //!< Applies forces to this voxel in the global coordinate system. Has no effect in any fixed degrees of freedom. @param xForce Force in the X direction in newtons.  @param yForce Force in the Y direction in newtons.  @param zForce Force in the Z direction in newtons. 
	CUDA_DEVICE void addForce(const TI_Vec3D<float>& force) {extForce += force;} //!< Convenience function for setExternalForce(float, float, float).
	CUDA_DEVICE void addMoment(const float xMoment, const float yMoment, const float zMoment) {extMoment += TI_Vec3D<float>(xMoment, yMoment, zMoment);}  //!< Applies moments to this voxel in the global coordinate system. All rotations according to the right-hand rule. Has no effect in any fixed degrees of freedom. @param xMoment Moment in the X axis rotation in newton-meters. @param yMoment Moment in the Y axis rotation in newton-meters. @param zMoment Moment in the Z axis rotation in newton-meters. 
	CUDA_DEVICE void addMoment(const TI_Vec3D<float>& moment) {extMoment += moment;} //!< Convenience function for setExternalMoment(float, float, float).

	CUDA_DEVICE void clearForce(){extForce = TI_Vec3D<float>();} //!< Clears all applied forces from this voxel.
	CUDA_DEVICE void clearMoment(){extMoment = TI_Vec3D<float>();} //!< Clears all applied moments from this voxel.

	CUDA_DEVICE void rotationChanged(); //called to keep cached quaternion rotation in sync

/* data */	
	dofObject dofFixed;
	
	TI_Vec3D<float> extForce, extMoment; //External force, moment applied to these voxels (N, N-m) if relevant DOF are unfixed
	TI_Vec3D<double> extTranslation, extRotation;
	TI_Quat3D<double> _extRotationQ; //cached quaternion rotation (pointer to only create if needed)

};

#endif // TI_EXTERNAL_H
