#include "TI_Material.h"

TI_Material::TI_Material( CVX_Material* p ):
linear(p->linear), E(p->E), sigmaYield(p->sigmaYield), sigmaFail(p->sigmaFail),
epsilonYield(p->epsilonYield), epsilonFail(p->epsilonFail), 
hd_strainData(p->strainData), hd_stressData(p->stressData), //hd_vector init in host, used for passing data to kernel. With syncVector() function, we use d_vector in kernel.
nu(p->nu),rho(p->rho),
alphaCTE(p->alphaCTE), muStatic(p->muStatic), muKinetic(p->muKinetic),
zetaInternal(p->zetaInternal), zetaGlobal(p->zetaGlobal), zetaCollision(p->zetaCollision),
extScale(p->extScale), _eHat(p->_eHat) {
	
}

CUDA_DEVICE TI_Material::TI_Material(float youngsModulus, float density)
{
	clear();
	rho = density;
	setModelLinear(youngsModulus);
	updateDerived();

}

CUDA_DEVICE TI_Material& TI_Material::operator=(const TI_Material& vIn)
{
	//error = vIn.error;
	//myName = vIn.myName;
	r = vIn.r;
	g = vIn.g;
	b = vIn.b;
	a = vIn.a;
	linear = vIn.linear;
	E = vIn.E;
	sigmaYield = vIn.sigmaYield;
	sigmaFail = vIn.sigmaFail;
	epsilonYield = vIn.epsilonYield;
	epsilonFail = vIn.epsilonFail;
	d_strainData = vIn.d_strainData;
	// hd_strainData = vIn.hd_strainData; //hd is only used for passing data to kernel. so no need to get the value.
	d_stressData = vIn.d_stressData;
	nu = vIn.nu;
	rho = vIn.rho;
	alphaCTE = vIn.alphaCTE;
	muStatic = vIn.muStatic;
	muKinetic = vIn.muKinetic;
	zetaInternal = vIn.zetaInternal;
	zetaGlobal = vIn.zetaGlobal;
	zetaCollision = vIn.zetaCollision;

	_eHat = vIn._eHat;

	return *this;
}

CUDA_DEVICE void TI_Material::clear()
{
	r = -1;
	g = -1;
	b = -1;
	a = -1;
	nu = 0.0f;
	rho = 1.0f;
	alphaCTE = 0.0f;
	muStatic = 0.0f;
	muKinetic = 0.0f;
	zetaInternal = 1.0f;
	zetaGlobal = 0.0f;
	zetaCollision = 0.0f;

	extScale=TI_Vec3D<>(1.0, 1.0, 1.0);

	setModelLinear(1.0);
	updateDerived();
}

CUDA_DEVICE float TI_Material::stress(float strain, float transverseStrainSum, bool forceLinear)
{
	//reference: http://www.colorado.edu/engineering/CAS/courses.d/Structures.d/IAST.Lect05.d/IAST.Lect05.pdf page 10
	if (isFailed(strain)) return 0.0f; //if a failure point is set and exceeded, we've broken!
	if (strain <= d_strainData[1] || linear || forceLinear){ //for compression/first segment and linear materials (forced or otherwise), simple calculation
		if (nu==0.0f) return E*strain;
		else return _eHat*((1-nu)*strain + nu*transverseStrainSum); 
//		else return eHat()*((1-nu)*strain + nu*transverseStrainSum); 
	}
	//the non-linear feature with non-zero poissons ratio is currently experimental
	int DataCount = modelDataPoints();
	for (int i=2; i<DataCount; i++){ //go through each segment in the material model (skipping the first segment because it has already been handled.
		if (strain <= d_strainData[i] || i==DataCount-1){ //if in the segment ending with this point (or if this is the last point extrapolate out) 
			float Perc = (strain-d_strainData[i-1])/(d_strainData[i]-d_strainData[i-1]);
			float basicStress = d_stressData[i-1] + Perc*(d_stressData[i]-d_stressData[i-1]);
			if (nu==0.0f) return basicStress;
			else { //accounting for volumetric effects
				float modulus = (d_stressData[i]-d_stressData[i-1])/(d_strainData[i]-d_strainData[i-1]);
				float modulusHat = modulus/((1-2*nu)*(1+nu));
				float effectiveStrain = basicStress/modulus; //this is the strain at which a simple linear stress strain line would hit this point at the definied modulus
				float effectiveTransverseStrainSum = transverseStrainSum*(effectiveStrain/strain);
				return modulusHat*((1-nu)*effectiveStrain + nu*effectiveTransverseStrainSum);
			}
		}
	}
	return 0.0f;
}

CUDA_DEVICE float TI_Material::strain(float stress)
{
	if (stress <= d_stressData[1] || linear) return stress/E; //for compression/first segment and linear materials (forced or otherwise), simple calculation

	int DataCount = modelDataPoints();
	for (int i=2; i<DataCount; i++){ //go through each segment in the material model (skipping the first segment because it has already been handled.
		if (stress <= d_stressData[i] || i==DataCount-1){ //if in the segment ending with this point (or if this is the last point extrapolate out) 
			float Perc = (stress-d_stressData[i-1])/(d_stressData[i]-d_stressData[i-1]);
			return d_strainData[i-1] + Perc*(d_strainData[i]-d_strainData[i-1]);
		}
	}
    return 0.0f;
}


CUDA_DEVICE float TI_Material::modulus(float strain)
{
	if (isFailed(strain)) return 0.0f; //if a failure point is set and exceeded, we've broken!
	if (strain <= d_strainData[1] || linear) return E; //for compression/first segment and linear materials, simple calculation

	int DataCount = modelDataPoints();
	for (int i=2; i<DataCount; i++){ //go through each segment in the material model (skipping the first segment because it has already been handled.
		if (strain <= d_strainData[i] || i==DataCount-1) return (d_stressData[i]-d_stressData[i-1])/(d_strainData[i]-d_strainData[i-1]); //if in the segment ending with this point
	}
	return 0.0f;
}

CUDA_DEVICE void TI_Material::setColor(int red, int green, int blue, int alpha)
{
	setRed(red);
	setGreen(green);
	setBlue(blue);
	setAlpha(alpha);
}

CUDA_DEVICE void TI_Material::setRed(int red)
{
	if (red>255) red=255;
	if (red<0) red=0;
	r = red;
}

CUDA_DEVICE void TI_Material::setGreen(int green)
{
	if (green>255) green=255;
	if (green<0) green=0;
	g = green;
}

CUDA_DEVICE void TI_Material::setBlue(int blue)
{
	if (blue>255) blue=255;
	if (blue<0) blue=0;
	b = blue;
}

CUDA_DEVICE void TI_Material::setAlpha(int alpha)
{
	if (alpha>255) alpha=255;
	if (alpha<0) alpha=0;
	a = alpha;
}

/*! The arrays are assumed to be of equal length.
The first data point is assumed to be [0,0] and need not be provided.
At least 1 non-zero data point must be provided.
The inital segment from [0,0] to the first strain and stress value is interpreted as young's modulus.
The slope of the stress/strain curve should never exceed this value in subsequent segments.
The last data point is assumed to represent failure of the material. The 0.2% offset method is used to calculate the yield point.

Restrictions on pStrainValues:
	- The values must be positive and increasing in order.
	- Strains are defined in absolute numbers according to delta l / L.

Restrictions on pStressValues:
	- The values must be positive and increasing in order.

Special cases:
	- 1 data point (linear): Yield and failure are assumed to occur simultaneously at the single data point.
	- 2 data points (bilinear): Yield is taken as the first data point, failure at the second.

*/
CUDA_DEVICE bool TI_Material::setModel(int dataPointCount, float* pStrainValues, float* pStressValues)
{
	assert(false); //not used.
	return false;
	// //Pre-checks
	// if (*pStrainValues==0 && *pStressValues==0) { //if first data point is 0,0, ignore it
	// 	pStrainValues++; //advance the pointers...
	// 	pStressValues++;
	// 	dataPointCount--; //decrement the count
	// }
	// if (dataPointCount<=0){
	// 	//error = "Not enough data points";
	// 	return false;
	// }
	// if (*pStrainValues<=0 || *pStressValues<=0){
	// 	//error = "First stress and strain data points negative or zero";
	// 	return false; 
	// }

	// //Copy the data into something more usable (and check for monotonically increasing)
	// TI_vector<float> tmpStrainData;
	// TI_vector<float> tmpStressData;
	// tmpStrainData.push_back(0); //add in the zero data point (required always)
	// tmpStressData.push_back(0);
	// float sweepStrain = 0.0f, sweepStress = 0.0f;
	// for (int i=0; i<dataPointCount; i++){
	// 	float thisStrain = *(pStrainValues+i); //grab the values
	// 	float thisStress = *(pStressValues+i);

	// 	if (thisStrain <= sweepStrain){
	// 		//error = "Out of order strain data";
	// 		return false;
	// 	}

	// 	if (thisStress <= sweepStress){
	// 		//error = "Stress data is not monotonically increasing";
	// 	}

	// 	if (i>0 && (thisStress-sweepStress)/(thisStrain-sweepStrain) > tmpStressData[0]/tmpStrainData[0]){
	// 		//error = "Slope of stress/strain curve should never exceed that of the first line segment (youngs modulus)";
	// 		return false;
	// 	}

	// 	sweepStrain = thisStrain;
	// 	sweepStress = thisStress;

	// 	tmpStrainData.push_back(thisStrain); //add to the temporary vector
	// 	tmpStressData.push_back(thisStress);
	// }

	// //at this point, we know we have valid data and will return true
	// strainData = tmpStrainData;
	// stressData = tmpStressData;
	// E=stressData[1]/strainData[1]; //youngs modulus is the inital slope
	// sigmaFail = stressData[stressData.size()-1]; //failure stress is the highest stress data point
	// epsilonFail = strainData[strainData.size()-1]; //failure strain is the highest strain data point
	// linear = (dataPointCount==1);

	// if (dataPointCount == 1 || dataPointCount == 2){ //linear or bilinear
	// 	sigmaYield = stressData[1];
	// 	epsilonYield = strainData[1];
	// }
	// else { //.2% (0.002) strain offset to find a good yield point...
	// 	setYieldFromData();
	// }

	// return updateDerived();
}

/*! Specified Young's modulus and failure stress must both be positive.
Yield stress is interpreted as identical to failure stress. If failure stress is not specified an arbitrary data point consistent with the specified Young's modulus is added to the model.
*/
CUDA_DEVICE bool TI_Material::setModelLinear(float youngsModulus, float failureStress)
{
	if (youngsModulus<=0){
		//error = "Young's modulus must be positive";
		return false;
	}
	if (failureStress != -1.0f && failureStress<=0){
		//error = "Failure stress must be positive";
		return false;
	}

	float tmpfailureStress = failureStress; //create a dummy failure stress if none was provided
	if (tmpfailureStress == -1) tmpfailureStress = 1000000;
	float tmpfailStrain = tmpfailureStress/youngsModulus;

	d_strainData.clear();
	d_stressData.clear();
	d_strainData.push_back(0.0f); //add in the zero data point (required always)
	d_stressData.push_back(0);
	d_strainData.push_back(tmpfailStrain);
	d_stressData.push_back(tmpfailureStress);

	linear=true;
	E=youngsModulus;
	sigmaYield = failureStress; //yield and failure are one in the same here.
	sigmaFail = failureStress;
	epsilonYield = (failureStress == -1) ? -1 : tmpfailStrain;
	epsilonFail = (failureStress == -1) ? -1 : tmpfailStrain;
	return updateDerived();
}

/*! Specified Young's modulus, plastic modulus, yield stress, and failure stress must all be positive.
Plastic modulus must be less than Young's modulus and failure stress must be greater than the yield stress.
*/
CUDA_DEVICE bool TI_Material::setModelBilinear(float youngsModulus, float plasticModulus, float yieldStress, float failureStress)
{
	if (youngsModulus<=0){
		//error = "Young's modulus must be positive";
		return false;
	}
	if (plasticModulus<=0 || plasticModulus>=youngsModulus){
		//error = "Plastic modulus must be positive but less than Young's modulus";
		return false;
	}
	if (yieldStress<=0){
		//error = "Yield stress must be positive";
		return false;
	}
	if (failureStress != -1.0f && failureStress <= yieldStress){
		//error = "Failure stress must be positive and greater than the yield stress";
		return false;
	}

	float yieldStrain = yieldStress / youngsModulus;
	float tmpfailureStress = failureStress; //create a dummy failure stress if none was provided
	if (tmpfailureStress == -1) tmpfailureStress = 3*yieldStress;

	float tM = plasticModulus;
	float tB = yieldStress-tM*yieldStrain; //y-mx=b
	float tmpfailStrain = (tmpfailureStress-tB)/tM; // (y-b)/m = x

	d_strainData.clear();
	d_strainData.push_back(0.0f); //add in the zero data point (required always)
	d_strainData.push_back(yieldStrain);
	d_strainData.push_back(tmpfailStrain);

	d_stressData.clear();
	d_stressData.push_back(0);
	d_stressData.push_back(yieldStress);
	d_stressData.push_back(tmpfailureStress);

	linear = false;
	E=youngsModulus;
	sigmaYield = yieldStress;
	sigmaFail = failureStress;
	epsilonYield = yieldStrain;
	epsilonFail = failureStress == -1.0f ? -1.0f : tmpfailStrain;
	return updateDerived();

}


CUDA_DEVICE bool TI_Material::setYieldFromData(float percentStrainOffset)
{
	sigmaYield = -1.0f; //assume we fail until we succeed.
	epsilonYield = -1.0f; //assume we fail until we succeed.

	float oM = E; //the offset line slope (y=Mx+B)
	float oB = (-percentStrainOffset/100*oM); //offset line intercept (100 factor turns percent into absolute

	int dataPoints = d_strainData.size()-1;
	for (int i=1; i<dataPoints-1; i++){
		float x1=d_strainData[i];
		float x2=d_strainData[i+1];
		float y1=d_stressData[i];
		float y2=d_stressData[i+1];

		float tM = (y2-y1)/(x2-x1); //temporary slope
		float tB = y1-tM*x1; //temporary intercept

		if (oM!=tM){ //if not parallel lines...
			float xIntersect = (tB-oB)/(oM-tM);
			if (xIntersect>x1 && xIntersect<x2){ //if intersects at this segment...
				float percentBetweenPoints = (xIntersect-x1)/(x2-x1);
				sigmaYield = y1+percentBetweenPoints*(y2-y1);
				epsilonYield = xIntersect;
				return true;
			}
		}
	}
	sigmaYield = sigmaFail;
	epsilonYield = epsilonFail;
	return false;
}

CUDA_DEVICE void TI_Material::setPoissonsRatio(float poissonsRatio)
{
	if (poissonsRatio < 0) poissonsRatio = 0;
	if (poissonsRatio >= 0.5 ) poissonsRatio = 0.5-FLT_EPSILON*2; //exactly 0.5 will still cause problems, but it can get very close.
	nu = poissonsRatio;
	updateDerived();
}

CUDA_DEVICE void TI_Material::setDensity(float density)
{
	if (density <= 0) density = FLT_MIN; //density of exactly 0 will cause problems, but can get as close as desired.
	rho = density;
	updateDerived();
}

CUDA_DEVICE void TI_Material::setStaticFriction(float staticFrictionCoefficient)
{
	if (staticFrictionCoefficient <= 0) staticFrictionCoefficient = 0;
	muStatic = staticFrictionCoefficient;
}

CUDA_DEVICE void TI_Material::setKineticFriction(float kineticFrictionCoefficient)
{
	if (kineticFrictionCoefficient <= 0) kineticFrictionCoefficient = 0;
	muKinetic = kineticFrictionCoefficient;
}

CUDA_DEVICE void TI_Material::setInternalDamping(float zeta)
{
	if (zeta <= 0) zeta = 0;
	zetaInternal = zeta;
}

CUDA_DEVICE void TI_Material::setGlobalDamping(float zeta)
{
	if (zeta <= 0) zeta = 0;
	zetaGlobal = zeta;
}

CUDA_DEVICE void TI_Material::setCollisionDamping(float zeta)
{
	if (zeta <= 0) zeta = 0;
	zetaCollision = zeta;
}

CUDA_DEVICE void TI_Material::setExternalScaleFactor(TI_Vec3D<double> factor)
{
	if (factor.x <= 0) factor.x = FLT_MIN;
	if (factor.y <= 0) factor.y = FLT_MIN;
	if (factor.z <= 0) factor.z = FLT_MIN;
	extScale = factor;
}

CUDA_DEVICE bool TI_Material::updateDerived() 
{
	_eHat = E/((1-2*nu)*(1+nu));

	//for (TI_vector<TI_Material*>::iterator it = dependentMaterials.begin(); it != dependentMaterials.end(); it++) (*it)->updateAll(); //update material properties of any that depend on this...
	// for (unsigned i=0;i<dependentMaterials.num_main;i++) {
	// 	dependentMaterials[i]->updateAll();
	// }
	return true;
}

CUDA_DEVICE void TI_Material::syncVectors()
{
	//hd_strainData -> d_strainData
	d_strainData.clear();
	d_strainData.push_back(0.0f);

	for (unsigned i=0;i<hd_strainData.size();i++) {
		d_strainData.push_back(hd_strainData[i]);
	}
	
	//hd_stressData -> d_stressData
	d_stressData.clear();
	d_stressData.push_back(0.0f);

	for (unsigned i=0;i<hd_stressData.size();i++) {
		d_stressData.push_back(hd_stressData[i]);
	}

	
}
