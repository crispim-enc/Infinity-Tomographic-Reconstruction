#include <stdint.h>

texture<char, 1> tex;
/*
Machine code for 
STEP: FORWARD PROJECTION 
CONFIGURATION: Multiple Kernel
PROJECTOR: Orthogonal 
DOI CORRECTION: FALSE 
DECAY CORRECTION: FALSE 
*/

__global__ void forward_projection_cdrf
(const int n, const int m, const int p, const int start_x, const int start_y, const int start_z,
    float* crystal_pitch_XY, float* crystal_pitch_Z, const float distance_between_array_pixel,
    int number_of_events, const int begin_event_gpu_limitation, const int end_event_gpu_limitation,
    float* a, float* a_normal, float* a_cf, float* b, float* b_normal, float* b_cf, float* c, float* c_normal,
    float* c_cf, float* d, float* d_normal, float* d_cf, float* sum_vor, const char* fov_cut_matrix, const float* im_old)
{
    const int shared_memory_size = 256;
    __shared__ float a_shared[shared_memory_size];
    __shared__ float b_shared[shared_memory_size];
    __shared__ float c_shared[shared_memory_size];
    __shared__ float d_shared[shared_memory_size];
    __shared__ float a_normal_shared[shared_memory_size];
    __shared__ float b_normal_shared[shared_memory_size];
    __shared__ float c_normal_shared[shared_memory_size];
    __shared__ float d_normal_shared[shared_memory_size];
    __shared__ float a_cf_shared[shared_memory_size];
    __shared__ float b_cf_shared[shared_memory_size];
    __shared__ float c_cf_shared[shared_memory_size];
    __shared__ float d_cf_shared[shared_memory_size];
    __shared__ float sum_vor_shared[shared_memory_size];
    __shared__ char fov_cut_matrix_shared[shared_memory_size];
    __shared__ float crystal_pitch_Z_shared[shared_memory_size];
    __shared__ float crystal_pitch_XY_shared[shared_memory_size];
  
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int e;

    float d2;
    float d2_normal;
    float orthogonal_factor;
    float y_distance;
    float z_distance;
//    float value_cf;
    float sum_vor_temp;
    int index;
    int x_t;
    int y_t;
    int z_t; 
    float max_distance_projector;

    const int number_events_max = end_event_gpu_limitation - begin_event_gpu_limitation;
    const float error_pixel = 0.0000f;
    if (threadIdx.x > shared_memory_size)
    {
        return;
    }
    if (threadId > number_events_max)
    {
        return;
    }


    e = threadId;
    int e_m = threadIdx.x;
    a_shared[e_m] = a[e];
    b_shared[e_m] = b[e];
    c_shared[e_m] = c[e];
    d_shared[e_m] = d[e];
    a_normal_shared[e_m] = a_normal[e];
    b_normal_shared[e_m] = b_normal[e];
    c_normal_shared[e_m] = c_normal[e];
    d_normal_shared[e_m] = d_normal[e];
    a_cf_shared[e_m] = a_cf[e];
    b_cf_shared[e_m] = b_cf[e];
    c_cf_shared[e_m] = c_cf[e];
    d_cf_shared[e_m] = d_cf[e];
    sum_vor_shared[e_m] = sum_vor[e];
    crystal_pitch_Z_shared[e_m] = crystal_pitch_Z[e];
    crystal_pitch_XY_shared[e_m] = crystal_pitch_XY[e];
     float FWHM = 8.0f;
     float FWHM_y = 1.45f/0.4;
       float FWHM_z = 1.45f/0.4;
     float current_distance;
     __syncthreads();
//    d2_normal = crystal_pitch_Z_shared[e_m] * sqrt(a_normal_shared[e_m] * a_normal_shared[e_m] + b_normal_shared[e_m] * b_normal_shared[e_m] + c_normal_shared[e_m] * c_normal_shared[e_m]);
//    d2 = crystal_pitch_XY_shared[e_m] * sqrt(a_shared[e_m] * a_shared[e_m] + b_shared[e_m] * b_shared[e_m] + c_shared[e_m] * c_shared[e_m]);
    d2_normal = crystal_pitch_Z_shared[e_m] ;//* sqrtf(a_normal_shared[e_m] * a_normal_shared[e_m] + b_normal_shared[e_m] * b_normal_shared[e_m] + c_normal_shared[e_m] * c_normal_shared[e_m]);
    d2 = crystal_pitch_XY_shared[e_m]; //* sqrtf(a_shared[e_m] * a_shared[e_m] + b_shared[e_m] * b_shared[e_m] + c_shared[e_m] * c_shared[e_m]);
//    FWHM_z = 0.0276*crystal_pitch_XY_shared[e_m]*crystal_pitch_XY_shared[e_m] + 0.281*crystal_pitch_XY_shared[e_m] + 0.517;
//    FWHM_y = 0.111*d2_normal*d2_normal/4 + 0.583*d2_normal/2+ 0.518;
//    FWHM_z = 0.111*d2*d2/4 + 0.583*d2/2+ 0.518;
    FWHM_y =d2_normal;
    FWHM_z =d2;
//    d2_cf = distance_between_array_pixel * sqrt(a_cf_shared[e_m] * a_cf_shared[e_m] + b_cf_shared[e_m] * b_cf_shared[e_m] + c_cf_shared[e_m] * c_cf_shared[e_m]);
//    max_distance_projector = sqrt(d2 * d2 + d2_normal * d2_normal + d2_cf * d2_cf);
//    max_distance_projector = sqrtf(d2 * d2 + d2_normal * d2_normal);
//    max_distance_projector = FWHM;
//    max_distance_projector = FWHM;

     for (int l = 0; l < n; l++)
    {
        x_t = l + start_x;

        for (int j = 0; j < m; j++)
        {
            y_t = j + start_y;

            for (int k = 0; k < p; k++)
            {
                index = k + j * p + l * p * m;

                z_t = k + start_z;
                if (fov_cut_matrix[index] != 0)
                {   y_distance = a_normal_shared[e_m] * x_t + b_normal_shared[e_m] * y_t + c_normal_shared[e_m] * z_t - d_normal_shared[e_m];
//                        y_distance = fabsf(y_distance);
//
                        if (fabsf(y_distance) <= d2_normal)
//                         { current_distance = sqrtf(y_distance * y_distance + z_distance * z_distance);
                         {
                    z_distance = a_shared[e_m] * x_t + b_shared[e_m] * y_t + c_shared[e_m] * z_t - d_shared[e_m];
//                    z_distance = fabsf(z_distance);
                    if (fabsf(z_distance) <= d2)
                    {
                         current_distance = sqrtf(y_distance * y_distance/(FWHM_y*FWHM_y) + z_distance * z_distance/(FWHM_z*FWHM_z));
//                            if (current_distance <= max_distance_projector)
//                            {
//                                orthogonal_factor = 1 - current_distance / max_distance_projector;
                                orthogonal_factor = 1 - current_distance;
                                    if (orthogonal_factor > 0.000f)
                                    {

                                    sum_vor_shared[e_m] += im_old[index] * orthogonal_factor;
                                    }
//                            }

                         // normalize the orthogonal factor to FWHM



                        }
                    }
                }
            }
        }
    }
    __syncthreads();
    sum_vor[e] = sum_vor_shared[e_m];

}