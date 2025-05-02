#include <stdint.h>
//texture<uint8_t, 1> tex;

__global__ void backprojection_cdrf
(float* sigma_y_square,  float* sigma_z_square, float* gaussian_y_fix_term, float* gaussian_z_fix_term,
float* acceptable_z_distance,  float* acceptable_y_distance,  float* invert2timesigma_y_square,
float* invert2timesigma_z_square, int dataset_number, int n, int m, int p, const float* crystal_pitch_XY, const float* crystal_pitch_Z,
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
    float width;
    float height;
    float distance;
    float distance_other;
    float solid_angle;
    float z_distance;
    float y_distance;
    float x_distance;
    float gaussian_x;
    float gaussian_y;
    float gaussian_z;
    bool shift_variant = false;


    if (idt >= n * m * p)
    {
        return;
    }


    adjust_coef_shared[i_s] = adjust_coef[idt];
    a_temp = A[idt];
    b_temp = B[idt];
    c_temp = C[idt];

    fov_cut_temp = fov_cut_matrix[idt];
__syncthreads();
//    float voxel_size = 0.5;
////    float sigma_x_square = 40/(voxel_size*voxel_size);
//    float sigma_y_square = 0.42*0.42/voxel_size;
//    float sigma_z_square = 0.42*0.42/voxel_size;
////    const float gaussian_x_fix_term = 1;
////    const float gaussian_y_fix_term = 1;
////    const float gaussian_z_fix_term = 1;
////    float gaussian_x_fix_term = 1 / rsqrtf(6.28 * sigma_x_square);
//    float gaussian_y_fix_term = 1 / rsqrtf(6.28 * sigma_y_square);
//    float gaussian_z_fix_term = 1 / rsqrtf(6.28 * sigma_z_square);
//    const float acceptable_z_distance = 11.09205*sigma_z_square;
//    const float acceptable_y_distance = 11.09205*sigma_y_square;
////    const float invert2sigma_x_square = 1 / (2 * sigma_x_square);
//    const float invert2timesigma_y_square = 1 / (2 * sigma_y_square);
//    const float invert2timesigma_z_square = 1 / (2 * sigma_z_square);


    for (int e = begin_event_gpu_limitation; e < end_event_gpu_limitation; e++)
    {
        if (fov_cut_temp != 0)
        {
//            z_distance = a_normal[e] * a_temp + b_normal[e] * b_temp + c_normal[e] * c_temp - d_normal[e];
//            y_distance = a[e] * a_temp + b[e] * b_temp + c[e] * c_temp - d[e];
//            x_distance = a_cf[e] * a_temp + b_cf[e] * b_temp + c_cf[e] * c_temp - d_cf[e];
            z_distance = a[e] * a_temp + b[e] * b_temp + c[e] * c_temp - d[e];
//            z_distance = fabsf(z_distance);

            if (fabsf(z_distance) <= acceptable_z_distance[e])
            {

                y_distance = a_normal[e] * a_temp + b_normal[e] * b_temp + c_normal[e] * c_temp - d_normal[e];
//                y_distance = fabsf(y_distance);
//                d2_normal = crystal_pitch_Z[e] * sqrt(a_normal[e] * a_normal[e] + b_normal[e] * b_normal[e] + c_normal[e] * c_normal[e]);
                if (fabsf(y_distance) <= acceptable_y_distance[e])
                {
//                {
//                    x_distance = a_cf[e] * a_temp + b_cf[e] * b_temp + c_cf[e] * c_temp - d_cf[e];
//                {
//                x_distance = a_cf[e] * a_temp + b_cf[e] * b_temp + c_cf[e] * c_temp - d_cf[e];
//                gaussian_x = gaussian_x_fix_term * __expf(-x_distance * x_distance * invert2sigma_x_square);

                if (sum_vor[e] != 0)
                {
                gaussian_y = gaussian_y_fix_term[e] * __expf(-y_distance * y_distance * invert2timesigma_y_square[e]);
                gaussian_z = gaussian_z_fix_term[e] * __expf(-z_distance * z_distance * invert2timesigma_z_square[e]);
                adjust_coef_shared[i_s] += (time_factor[e] * gaussian_y *gaussian_z) / (sum_vor[e]);
                }
//                x_distance = fabsf(x_distance);

//                 if (shift_variant)
//                    {    d2_normal = crystal_pitch_Z[e] * sqrt(a_normal[e] * a_normal[e] + b_normal[e] * b_normal[e] + c_normal[e] * c_normal[e]);
//
//                        d2_cf = distance_between_array_pixel * sqrt(a_cf[e] * a_cf[e] + b_cf[e] * b_cf[e] + c_cf[e] * c_cf[e]);
//                        sigma_x = distance_between_array_pixel/2.35;
//                        sigma_y = crystal_pitch_Z[e]/2.35;
//                        sigma_z = crystal_pitch_XY[e]/2.35;
//                    }



                }
//
//            }
//            }

            }
        }

    }

    adjust_coef[idt] = adjust_coef_shared[i_s];
    __syncthreads();



}