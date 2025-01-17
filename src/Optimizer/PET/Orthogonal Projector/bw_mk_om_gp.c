#include <stdint.h>
texture<uint8_t, 1> tex;

__global__ void backprojection
(int dataset_number, int n, int m, int p, const float crystal_pitch_XY, const float crystal_pitch_Z,
    const float distance_between_array_pixel, int number_of_events, const int begin_event_gpu_limitation,
    const int end_event_gpu_limitation, const float* a, const float* a_normal, const float* a_cf, const float* b,
    const float* b_normal, const float* b_cf, const float* c, const float* c_normal, const float* c_cf, const float* d, const float* d_normal,
    const float* d_cf, const short* A, const short* B, const short* C, float* adjust_coef, float* sum_vor,
    char* fov_cut_matrix, float* time_factor)
{
    extern __shared__ float adjust_coef_shared[];
    int idt = blockIdx.x * blockDim.x + threadIdx.x;


    float d2;
    float d2_normal;
    float d2_cf;
    float normal_value;
    float value;
    float value_cf;
    short a_temp;
    short b_temp;
    short c_temp;
    char fov_cut_temp;
    int i_s = threadIdx.x;

    if (idt > n * m * p)
    {
        return;
    }

    __syncthreads();
    adjust_coef_shared[i_s] = adjust_coef[idt];
    a_temp = A[idt];
    b_temp = B[idt];
    c_temp = C[idt];
    fov_cut_temp = fov_cut_matrix[idt];



    for (int e = begin_event_gpu_limitation; e < end_event_gpu_limitation; e++)
    {
        if (fov_cut_temp != 0)
        {
            normal_value = a_normal[e] * a_temp + b_normal[e] * b_temp + c_normal[e] * c_temp - d_normal[e];
            d2_normal = crystal_pitch_XY * sqrt(a_normal[e] * a_normal[e] + b_normal[e] * b_normal[e] + c_normal[e] * c_normal[e]);

            if (normal_value < d2_normal && normal_value >= -d2_normal)
            {
                value = a[e] * a_temp + b[e] * b_temp + c[e] * c_temp - d[e];
                d2 = crystal_pitch_Z * sqrt(a[e] * a[e] + b[e] * b[e] + c[e] * c[e]);


                if (value < d2 && value >= -d2)
                {
                    value_cf = a_cf[e] * a_temp + b_cf[e] * b_temp + c_cf[e] * c_temp - d_cf[e];
                    d2_cf = distance_between_array_pixel * sqrt(a_cf[e] * a_cf[e] + b_cf[e] * b_cf[e] + c_cf[e] * c_cf[e]);


                    if (value_cf >= -d2_cf && value_cf < d2_cf)
                    {

                        adjust_coef_shared[i_s] += (1 - (sqrt(value * value + normal_value * normal_value + value_cf * value_cf)) / sqrt(d2 * d2 + d2_normal * d2_normal + d2_cf * d2_cf)) / (sum_vor[e] * time_factor[e]);

                    }




                }
            }

        }

    }

    adjust_coef[idt] = adjust_coef_shared[i_s];
    __syncthreads();



}