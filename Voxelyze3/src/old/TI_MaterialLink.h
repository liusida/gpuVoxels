#if !defined(TI_MATERIAL_LINK_H)
#define TI_MATERIAL_LINK_H

#include "TI_Utils.h"
#include "VX_MaterialLink.h"
#include "TI_MaterialVoxel.h"

class TI_MaterialLink : public TI_MaterialVoxel {
public:
	TI_MaterialLink(CVX_MaterialLink* p, cudaStream_t stream);
	~TI_MaterialLink();

	CUDA_DEVICE TI_MaterialLink(TI_MaterialVoxel* mat1, TI_MaterialVoxel* mat2); //!< Creates a link material from the two specified voxel materials. The order is unimportant. @param[in] mat1 voxel material on one side of the link. @param[in] mat2 voxel material on the other side of the link.
	CUDA_DEVICE TI_MaterialLink(const TI_MaterialLink& VIn) {*this = VIn;} //!< Copy constructor

	CUDA_DEVICE virtual TI_MaterialLink& operator=(const TI_MaterialLink& VIn); //!< Equals operator

	CUDA_DEVICE virtual bool updateAll(); //!< Updates and recalculates eveything possible (used by inherited classed when material properties have changed)
	CUDA_DEVICE virtual bool updateDerived(); //!< Updates all the derived quantities cached as member variables for this and derived classes. (Especially if density, size or elastic modulus changes.)

/* data */
	TI_MaterialVoxel *vox1Mat = NULL; //!< Constituent material 1 from one voxel
	TI_MaterialVoxel *vox2Mat = NULL; //!< Constituent material 2 from the other voxel

	float _a1; //!< Cached a1 beam constant.
	float _a2; //!< Cached a2 beam constant.
	float _b1; //!< Cached b1 beam constant.
	float _b2; //!< Cached b2 beam constant.
	float _b3; //!< Cached b3 beam constant.
	float _sqA1; //!< Cached sqrt(a1) constant for damping calculations.
	float _sqA2xIp; //!< Cached sqrt(a2*L*L/6) constant for damping calculations.
	float _sqB1; //!< Cached sqrt(b1) constant for damping calculations.
	float _sqB2xFMp; //!< Cached sqrt(b2*L/2) constant for damping calculations.
	float _sqB3xIp; //!< Cached sqrt(b3*L*L/6) constant for damping calculations.
	
};

#endif // TI_MATERIAL_LINK_H
