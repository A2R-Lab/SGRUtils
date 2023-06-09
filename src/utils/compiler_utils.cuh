#ifndef UTILS_COMP_CUH
#define UTILS_COMP_CUH

// helpful compiler trick
// https://stackoverflow.com/questions/32014839/how-to-use-a-cuda-class-header-file-in-both-cpp-and-cuda-modules
#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

#endif