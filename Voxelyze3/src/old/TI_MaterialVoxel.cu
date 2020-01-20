#include "TI_MaterialVoxel.h"

TI_MaterialVoxel::TI_MaterialVoxel( CVX_MaterialVoxel *p, cudaStream_t stream ):
TI_Material( (CVX_Material*) p, stream ),
nomSize(p->nomSize), gravMult(p->gravMult),_mass(p->_mass),
_massInverse(p->_massInverse), _sqrtMass(p->_sqrtMass), _firstMoment(p->_firstMoment),
_momentInertia(p->_momentInertia), _momentInertiaInverse(p->_momentInertiaInverse),
_2xSqMxExS(p->_2xSqMxExS), _2xSqIxExSxSxS(p->_2xSqIxExSxSxS) {

}

CUDA_DEVICE TI_MaterialVoxel::TI_MaterialVoxel(float youngsModulus, float density, double nominalSize) : TI_Material(youngsModulus, density)
{
	initialize(nominalSize);
}

CUDA_DEVICE TI_MaterialVoxel::TI_MaterialVoxel(const TI_Material& mat, double nominalSize) : TI_Material(mat)
{
	initialize(nominalSize);
}

CUDA_DEVICE void TI_MaterialVoxel::initialize(double nominalSize)
{
	nomSize = nominalSize;
	gravMult = 0.0f;
	updateDerived();
}

CUDA_DEVICE TI_MaterialVoxel& TI_MaterialVoxel::operator=(const TI_MaterialVoxel& vIn)
{
	TI_Material::operator=(vIn); //set base TI_Material class variables equal

	nomSize=vIn.nomSize;
	gravMult=vIn.gravMult;
	_eHat = vIn._eHat;
	_mass=vIn._mass;
	_massInverse=vIn._massInverse;
	_sqrtMass=-vIn._sqrtMass;
	_firstMoment=vIn._firstMoment;
	_momentInertia=vIn._momentInertia;
	_momentInertiaInverse=vIn._momentInertiaInverse;
	_2xSqMxExS=vIn._2xSqMxExS;
	_2xSqIxExSxSxS=vIn._2xSqIxExSxSxS;

	return *this;
}

CUDA_DEVICE bool TI_MaterialVoxel::updateDerived() 
{
	TI_Material::updateDerived(); //update base TI_Material class derived variables

	double volume = nomSize*nomSize*nomSize;
	_mass = (float)(volume*rho); 
	_momentInertia = (float)(_mass*nomSize*nomSize / 6.0f); //simple 1D approx
	_firstMoment = (float)(_mass*nomSize / 2.0f);

	if (volume==0 || _mass==0 || _momentInertia==0){
		_massInverse = _sqrtMass = _momentInertiaInverse = _2xSqMxExS = _2xSqIxExSxSxS = 0.0f; //zero everything out
		return false;
	}


	_massInverse = 1.0f / _mass;
	_sqrtMass = sqrt(_mass);
	_momentInertiaInverse = 1.0f / _momentInertia;
	_2xSqMxExS = (float)(2.0f*sqrt(_mass*E*nomSize));
	_2xSqIxExSxSxS = (float)(2.0f*sqrt(_momentInertia*E*nomSize*nomSize*nomSize));

	return true;
}


CUDA_DEVICE bool TI_MaterialVoxel::setNominalSize(double size)
{
	if (size <= 0) size = FLT_MIN;
	nomSize=size;
	return updateDerived(); //update derived quantities
}