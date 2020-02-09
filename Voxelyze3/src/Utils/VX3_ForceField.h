#if !defined(VX3_FORCEFIELD_H)
#define VX3_FORCEFIELD_H

#include "VX3_MathTree.h"

struct VX3_ForceField {
    VX3_MathTreeToken token_x_prime[1024];
    VX3_MathTreeToken token_y_prime[1024];
    VX3_MathTreeToken token_z_prime[1024];
    VX3_ForceField() {
        token_x_prime[0].op = mtCONST;
        token_x_prime[0].value = 0;
        token_x_prime[1].op = mtEND;
        token_y_prime[0].op = mtCONST;
        token_y_prime[0].value = 0;
        token_y_prime[1].op = mtEND;
        token_z_prime[0].op = mtCONST;
        token_z_prime[0].value = 0;
        token_z_prime[1].op = mtEND;
    }
    bool validate() {
        return VX3_MathTree::validate(token_x_prime) &&
               VX3_MathTree::validate(token_y_prime) &&
               VX3_MathTree::validate(token_z_prime);
    }
    __device__ __host__ double x_prime(double x, double y, double z, double t) {
        return VX3_MathTree::eval(x, y, z, t, token_x_prime);
    }
    __device__ __host__ double y_prime(double x, double y, double z, double t) {
        return VX3_MathTree::eval(x, y, z, t, token_y_prime);
    }
    __device__ __host__ double z_prime(double x, double y, double z, double t) {
        return VX3_MathTree::eval(x, y, z, t, token_z_prime);
    }
};

#endif // VX3_FORCEFIELD_H
