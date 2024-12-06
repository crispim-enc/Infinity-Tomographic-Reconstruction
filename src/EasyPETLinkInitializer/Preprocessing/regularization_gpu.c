#include <stdint.h>

__global__ void sinogram_regularization
(int dataset_number)
{
//    extern __shared__ float adjust_coef_shared[];
    const int shared_memory_size = 128;
    __shared__ float adjust_coef_shared[shared_memory_size];
    __shared__ short a_temp[shared_memory_size];
    __shared__ short b_temp[shared_memory_size];
    __shared__ short c_temp[shared_memory_size];
    __shared__ char fov_cut_temp[shared_memory_size];
    int idt = blockIdx.x * blockDim.x + threadIdx.x;


    float d2;
    float d2_normal;
    float d2_cf;
    float normal_value;
    float value;
    float value_cf;
//    short a_temp;
//    short b_temp;
//    short c_temp;
//    char fov_cut_temp;
    int i_s = threadIdx.x;
    float width;
    float height;
    float distance;
    float distance_other;
    float solid_angle;

    if (idt >= n * m * p)
    {
        return;
    }

    __syncthreads();
    for (int )
    adjust_coef_shared[i_s] = adjust_coef[idt];
    a_temp[i_s] = A[idt];
    b_temp[i_s] = B[idt];
    c_temp[i_s] = C[idt];

    fov_cut_temp[i_s] = fov_cut_matrix[idt];

    for (int e = begin_event_gpu_limitation; e < end_event_gpu_limitation; e++)
    {

    }
      __syncthreads();
    adjust_coef[idt] = adjust_coef_shared[i_s];




}