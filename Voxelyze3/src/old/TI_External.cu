#include "TI_External.h"

TI_External::TI_External( CVX_External* p ) :
dofFixed(p->dofFixed), extForce(p->extForce), extMoment(p->extMoment),
extTranslation(p->extTranslation), extRotation(p->extRotation), 
_extRotationQ(p->_extRotationQ) {

}

CUDA_DEVICE TI_External::TI_External() 
{
	reset();
}

CUDA_DEVICE TI_External& TI_External::operator=(const TI_External& eIn)
{
	dofFixed = eIn.dofFixed;
	extForce = eIn.extForce;
	extMoment = eIn.extMoment;
	extTranslation = eIn.extTranslation;
	extRotation = eIn.extRotation;
	rotationChanged();
	return *this;
}

CUDA_DEVICE void TI_External::reset()
{
	dofFixed=0;
	extForce = extMoment = TI_Vec3D<float>();
	extTranslation = TI_Vec3D<double>();
	extRotation = TI_Vec3D<double>();
	rotationChanged();
}


CUDA_DEVICE void TI_External::setFixed(bool xTranslate, bool yTranslate, bool zTranslate, bool xRotate, bool yRotate, bool zRotate)
{
	dofFixed = dof(xTranslate, yTranslate, zTranslate, xRotate, yRotate, zRotate);
	extTranslation = extRotation = TI_Vec3D<double>(); //clear displacements
}

CUDA_DEVICE void TI_External::setDisplacement(dofComponent dof, double displacement)
{
	dofSet(dofFixed, dof, true);
	if (displacement != 0.0f){
		if (dof & X_TRANSLATE) extTranslation.x = displacement;
		if (dof & Y_TRANSLATE) extTranslation.y = displacement;
		if (dof & Z_TRANSLATE) extTranslation.z = displacement;
		if (dof & X_ROTATE) extRotation.x = displacement;
		if (dof & Y_ROTATE)	extRotation.y = displacement;
		if (dof & Z_ROTATE) extRotation.z = displacement;
	}

	rotationChanged();
}

CUDA_DEVICE void TI_External::setDisplacementAll(const TI_Vec3D<double>& translation, const TI_Vec3D<double>& rotation)
{
	dofSetAll(dofFixed, true);
	extTranslation = translation;
	extRotation = rotation;

	rotationChanged();
}

CUDA_DEVICE void TI_External::clearDisplacement(dofComponent dof)
{
	dofSet(dofFixed, dof, false);

	if (dof & X_TRANSLATE) extTranslation.x = 0.0;
	if (dof & Y_TRANSLATE) extTranslation.y = 0.0;
	if (dof & Z_TRANSLATE) extTranslation.z = 0.0;
	if (dof & X_ROTATE) extRotation.x = 0.0;
	if (dof & Y_ROTATE)	extRotation.y = 0.0;
	if (dof & Z_ROTATE) extRotation.z = 0.0;

	rotationChanged();
}

CUDA_DEVICE void TI_External::clearDisplacementAll()
{
	dofSetAll(dofFixed, false);
	extTranslation = TI_Vec3D<double>();
	extRotation = TI_Vec3D<double>();

	rotationChanged();
}

CUDA_DEVICE void TI_External::rotationChanged()
{
	if (extRotation != TI_Vec3D<double>()){
		_extRotationQ = TI_Quat3D<double>(extRotation);
	}
	else { //rotation is zero in all axes
		_extRotationQ = TI_Quat3D<double>();
	}
}




