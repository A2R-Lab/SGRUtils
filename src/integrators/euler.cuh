#ifndef SGRUTILS_EULER_CUH
#define SGRUTILS_EULER_CUH

#include <stdio.h>

namespace sgrutils {

__host__
__device__
void doubleLoopVals(int *starty, int *dy, int *startx, int *dx){
    #ifdef __CUDA_ARCH__
        *starty = threadIdx.y;
        *dy = blockDim.y;
        *startx = threadIdx.x;
        *dx = blockDim.x;
    #else
        *starty = 0;
        *dy = 1;
        *startx = 0;
        *dx = 1;
    #endif
}

__host__
__device__
void singleLoopVals(int *start, int *delta){
    #ifdef __CUDA_ARCH__
        *start = threadIdx.x + threadIdx.y*blockDim.x;
        *delta = blockDim.x*blockDim.y;
    #else
        *start = 0;
        *delta = 1;
    #endif
}

template <typename T>
__host__
__device__
T dqdd2dxd(T *dqdd, int r, int c, int num_pos){
    if (r < num_pos) {
        if (r + num_pos == c) {
            return static_cast<T>(1);
        } else {
            return static_cast<T>(0);
        }
    } else {
        int index = (c - 1) * num_pos + r;
        return dqdd[index];
    }
}

template <typename T>
__host__
__device__ __forceinline__
void _integrator(T * s_next_state, T * s_x, T *s_qdd, T dt, int num_pos){
    int start, delta;
    singleLoopVals(&start,&delta);

    // the euler rule defines next state as
    // [q', qd'] = [q, qd] + dt * [qd, qdd]

    // ind will iterate over the positions
    for (int ind = start; ind < num_pos; ind += delta){

        int position_index = ind;
        int velocity_index = ind + num_pos; // velocity corresponding to this position in state vector
        int acceleration_index = ind; // acceleration corresponding to this position in state vector

        s_next_state[position_index] = s_x[position_index] + dt*s_x[velocity_index];
        s_next_state[velocity_index] = s_x[velocity_index] + dt*s_qdd[acceleration_index];
    }
}

/*
Euler integrator computes next state as [q', qd'] = [q, qd] + dt * [qd, qdd]

We want to compute the gradient of this function, so we need the partial derivatives:

Return the partial derivative matricies:
    A = [[dq'/dq, dq'/dqd],    and B = [[dq/du],
         [dqd'/dq, dqd'/dqd]]           [dqd/du]]

so the result we want becomes:

 [ 0       ; eye     ; 0
   dqdd/dq ; dqdd/dv ; dqdd/du ]
*/ 
template <typename T>
__host__
__device__ __forceinline__
void _integratorGradient(T * ABk, T *s_dqdd, T dt, int dim_AB_r, int dim_AB_c, int num_pos){
    int starty, dy, startx, dx;
    doubleLoopVals(&starty,&dy,&startx,&dx);
    // ky will determine the column
    // column 1 -> partial derivatives with respect to change in q
    // column 2 -> partial derivatives with respect to change in q dot
    // column 3 -> partial derivatives with respect to chnage in u
    #pragma unroll
    for (int ky = starty; ky < dim_AB_c; ky += dy){ 
        // kx will determine the row
        // row 1 -> q
        // row 2 -> q dot
        #pragma unroll
        for (int kx = startx; kx < dim_AB_r; kx += dx){ 
            int offset = static_cast<T>(ky == kx ? 1 : 0);
            T val = offset + dt * dqdd2dxd<T>(s_dqdd,kx,ky, num_pos=num_pos);
            ABk[ky*dim_AB_r + kx] = val;
        }
    }
}

} // namespace sgrutils

#endif // SGRUTILS_EULER_CUH
