#include "Voxelyze.h"

int main() {
    CVoxelyze Vx(0.005); // 5mm voxels
    CVX_Material *pMaterial = Vx.addMaterial(
        1000000,
        1000); // A material with stiffness E=1MPa and density 1000Kg/m^3
    CVX_Voxel *Voxel1 =
        Vx.setVoxel(pMaterial, 0, 0, 0); // Voxel at index x=0, y=0. z=0
    CVX_Voxel *Voxel2 =
        Vx.setVoxel(pMaterial, 1, 0, 0); // Voxel at index x=0, y=0. z=0
    Voxel2->pos.x -= 0.0001;
    for (int i = 0; i < 100; i++)
        Vx.doTimeStep(); // simulates 100 timesteps
}