#ifndef SGRUTILS_UTILS_CUH
#define SGRUTILS_UTILS_CUH

/*
    Cuda utility functions
*/
#include <cuda_runtime.h>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
// #include <iiwa-grid.cuh>

// Move these outside of any namespace
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

namespace sgrutils {

template <typename T>
__device__ T* shared_memory_proxy()
{
    // __align__(sizeof(T)) -- this will break if multiple Ts chosen
    extern __shared__ unsigned char memory[];
    return reinterpret_cast<T*>(memory);
}

__device__ void sleepKernel()
{
    unsigned long long int start = clock64(); // get the start time
    unsigned long long int delay = 1000000000; // sleep for 1 second
    
    while (clock64() < start + delay) {
        // busy-wait
    }
    
    printf("Thread %d woke up after sleeping for 1 second.\n", threadIdx.x);
}

template <typename T>
__device__
__host__
void printMatrixColumnMajor(T * matrix, int rows, int cols) {
    for(int i=0; i<rows; i++) {
        for(int j=0; j<cols; j++) {
            printf("%f ", matrix[j*rows + i]);
        }
        printf("\n");
    }
}

template <typename T>
__device__
__host__
void printMatrixColumnMajor2(T * matrix, int rows, int cols, int gap) {
    for(int i=0; i<rows; i++) {
        for(int j=0; j<cols; j++) {
            printf("%f ", matrix[j*rows + i + j*gap]);
        }
        printf("\n");
    }
}

// prints col major order matrix to file in numpy format
template <typename T> __device__ __host__ void printMatrixColMajorNumpy(T *matrix, int rows, int cols)
{
    printf("[");
    for (int i = 0; i < rows; ++i)
    {
        printf("[");
        for (int j = 0; j < cols; ++j)
        {
            // Assuming T can be a float, adjust the format specifier if T changes
            printf("%d", matrix[j * rows + i]);
            if (j < cols - 1)
                printf(", ");
        }
        printf("]");
        if (i < rows - 1)
            printf(",\n");
    }
    printf("]\n");
}

template <typename T>
__device__ void debugPrint(int threadIdxToPrint, int blockIdxToPrint, const char *message, T *matrix, int rows,
                           int cols, bool isMatrix = true, int index = -1)
{
    if (DEBUG_SWITCH)
    {
        __syncthreads();
        if (threadIdx.x == threadIdxToPrint && blockIdx.x == blockIdxToPrint)
        {
            if (index >= 0)
            {
                printf("index: %d\n", index);
            }
            printf("%s\n", message);
            if (isMatrix)
            {
                sgrutils::printMatrixColumnMajor(matrix, rows, cols);
            }
            else
            {
                for (int i = 0; i < rows; i++)
                {
                    printf("%d, ", matrix[i]);
                }
            }
            printf("\n");
        }
        __syncthreads();
    }
}

template <typename T>
__device__
__host__
void printMatrix(T * matrix, int rows, int cols) {
    for(int i=0; i<rows; i++) {
        for(int j=0; j<cols; j++) {
            printf("%f ", matrix[i*cols + j]);
        }
        printf("\n");
    }
}

// copies a matrix (optionally scales by alpha)
template <typename T, int M, int N>
__host__ __device__ __forceinline__
void copyMat(T *dst, T *src, int ld_dst, int ld_src, T alpha = 1.0){
    int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
    #pragma unroll
    for (int ky = starty; ky < N; ky += dy){
        #pragma unroll
        for (int kx = startx; kx < M; kx += dx){
            dst[kx + ld_dst*ky] = alpha*src[kx + ld_src*ky];
        }
    }
}

// loads a matrix into shared memory
// special case of copyMat, assumes shared memory is of size m*n and original matrix is on disk size ld*n
template <typename T, int M, int N>
__host__  __device__ __forceinline__
void loadMatToShared(T *dst, T *src, int ld){
    copyMat<T,M,N>(dst, src, M, ld);
}

// loads and regularizes a matrix
template <typename T, int M, int N>
__host__  __device__ __forceinline__
void loadAndReg(T *dst, T *src, int ld_dst, int ld_src, T reg){
    int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
    #pragma unroll
    for (int ky = starty; ky < N; ky += dy){
        #pragma unroll
        for (int kx = startx; kx < M; kx += dx){
            dst[kx + ld_dst*ky] = src[kx + ld_src*ky] + (kx == ky ? reg : static_cast<T>(0));
        }
    }
}
// loads a and regularizes a matrix into shared memory
// special case of copyMat, assumes shared memory is of size m*n and original matrix is on disk size ld*n
template <typename T, int M, int N>
__host__ __device__ __forceinline__
void loadAndRegToShared(T *dst, T *src, int ld, T reg){
    loadAndReg<T,M,N>(dst, src, N, ld, reg);
}

// loads the identiy matrix into a variable
template <typename T, int M, int N>
__host__  __device__ __forceinline__
void loadIdentity(T *A, int ld_A){
    int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
    #pragma unroll
    for (int ky = starty; ky < N; ky += dy){
        #pragma unroll
        for (int kx = startx; kx < M; kx += dx){
            A[ky*ld_A + kx] = static_cast<T>(kx == ky ? 1 : 0);
        }
    }
}

__host__  __device__
void parse_csv_to_int_vec(std::vector<int> * parsed_csv, std::string path, int rows, int cols) {
    std::ifstream data(path);
    std::string line;

    while(std::getline(data,line))
    {
        std::stringstream lineStream(line);
        std::string cell;
        while(std::getline(lineStream,cell,','))
        {
            (*parsed_csv).push_back(stoi(cell));
        }
    }
}

void parse_csv_to_float_vec(std::vector<float> * parsed_csv, std::string path, int rows, int cols) {
    std::ifstream data(path);
    std::string line;

    while(std::getline(data,line))
    {
        std::stringstream lineStream(line);
        std::string cell;
        while(std::getline(lineStream,cell,','))
        {
            (*parsed_csv).push_back(stof(cell));
        }
    }
}

void parse_csv_to_double_vec(std::vector<double> * parsed_csv, std::string path) {
    std::ifstream data(path);
    std::string line;

    while(std::getline(data,line))
    {
        std::stringstream lineStream(line);
        std::string cell;
        while(std::getline(lineStream,cell,','))
        {
            (*parsed_csv).push_back(stod(cell));
        }
    }
}

template <typename T>
__host__
void parse_csv_to_vec(std::vector<std::vector<T>> * parsed_csv, std::string path) {
    std::ifstream data(path);
    std::string line;

    while (std::getline(data, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<T> row;  // Temporary vector for the current row
        while (std::getline(lineStream, cell, ',')) {
            std::istringstream cellStream(cell);
            T value;
            cellStream >> value;  // Convert cell to type T
            row.push_back(value);
        }
        parsed_csv->push_back(row);  // Add the row to the 2D vector
    }
}

template <typename T>
__device__
void diagonalize_vector(u_int32_t N, T * arr, T * out) {
    for (int i = threadIdx.x; i < N * N; i += blockDim.x) {
        out[i] = (i/N == i%N) ? arr[i/N] : 0;
    } 
}

// Function to write trajectory data to CSV file
template <typename T>
__host__
void write_trajectory_to_csv(const std::vector<T>& trajectory, const std::string& filename, 
                           size_t num_timesteps, size_t data_size, bool include_trailing_comma = true) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        fprintf(stderr, "Error: Could not open file %s for writing\n", filename.c_str());
        return;
    }
    
    for (size_t t = 0; t < num_timesteps; t++) {
        for (size_t i = 0; i < data_size; i++) {
            out << trajectory[t * data_size + i];
            if (include_trailing_comma || i < data_size - 1) {
                out << ",";
            }
        }
        out << "\n";
    }
    out.close();
}

// Function to write multiple trajectories to CSV files
template <typename T>
__host__
void write_trajectories_to_csv(const std::vector<T>& x_trajectory, const std::vector<T>& u_trajectory, 
                              const std::vector<T>& ee_trajectory, size_t num_timesteps, 
                              size_t state_size, size_t control_size, size_t ee_pose_size,
                              const std::string& file_identifier,
                              const std::string& output_dir = "results/") {
    // Write state trajectory
    write_trajectory_to_csv(x_trajectory, output_dir + "x_trajectory_" + file_identifier + ".csv", num_timesteps, state_size);

    // Write control trajectory (one less timestep)
    write_trajectory_to_csv(u_trajectory, output_dir + "u_trajectory_" + file_identifier + ".csv", num_timesteps - 1, control_size);

    // Write end-effector trajectory
    write_trajectory_to_csv(ee_trajectory, output_dir + "ee_trajectory_" + file_identifier + ".csv", num_timesteps, ee_pose_size);

    printf("Trajectories saved to CSV files in %s\n", output_dir.c_str());
}

} // namespace sgrutils

#endif // SGRUTILS_UTILS_CUH