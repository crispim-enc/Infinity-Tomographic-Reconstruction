#include <stdint.h>
texture<uint8_t, 1> tex;

__global__ void backprojection_cdrf
(int dataset_number, int n, int m, int p, const float* crystal_pitch_XY, const float* crystal_pitch_Z,
    const float distance_between_array_pixel, int number_of_events, const int begin_event_gpu_limitation,
    const int end_event_gpu_limitation, const float* a, const float* a_normal, const float* a_cf, const float* b,
    const float* b_normal, const float* b_cf, const float* c, const float* c_normal, const float* c_cf, const float* d, const float* d_normal,
    const float* d_cf, const short* A, const short* B, const short* C, float* adjust_coef, float* sum_vor,
    char* fov_cut_matrix, float* time_factor)
{
    extern __shared__ float adjust_coef_shared[];
    int idt = blockIdx.x * blockDim.x + threadIdx.x;

    float orthogonal_factor;
    float d2;
    float d2_normal;
    float y_distance;
    float z_distance;

    short a_temp;
    short b_temp;
    short c_temp;
    char fov_cut_temp;
    int i_s = threadIdx.x;


    if (idt >= n * m * p)
    {
        return;
    }


    adjust_coef_shared[i_s] = adjust_coef[idt];
    __syncthreads();
    a_temp = A[idt];
    b_temp = B[idt];
    c_temp = C[idt];

    fov_cut_temp = fov_cut_matrix[idt];
    float FWHM = 8.0f;
    float FWHM_y = 1.45f/0.4;
    float FWHM_z = 1.45f/0.4;
    float current_distance;
    for (int e = begin_event_gpu_limitation; e < end_event_gpu_limitation; e++)
    {
        if (fov_cut_temp != 0)
        {
            y_distance = a_normal[e] * a_temp + b_normal[e] * b_temp + c_normal[e] * c_temp - d_normal[e];
            d2_normal = crystal_pitch_Z[e]; //* sqrtf(a_normal[e] * a_normal[e] + b_normal[e] * b_normal[e] + c_normal[e] * c_normal[e]);
             if (fabsf(y_distance) <= d2_normal)
             {
                z_distance = a[e] * a_temp + b[e] * b_temp + c[e] * c_temp - d[e];
                d2 = crystal_pitch_XY[e]; //* sqrtf(a[e] * a[e] + b[e] * b[e] + c[e] * c[e]);

                if (fabsf(z_distance) <= d2)
                {
//                     current_distance = sqrtf(z_distance * z_distance + y_distance * y_distance);
//                    FWHM_z = 0.0276*crystal_pitch_XY[e]*crystal_pitch_XY[e] + 0.281*crystal_pitch_XY[e]+ 0.517;

//                    FWHM_z = 0.111*d2*d2/4 + 0.583*d2/2+ 0.518;
//                    FWHM_y = 0.111*d2_normal*d2_normal/4 + 0.583*d2_normal/2+ 0.518;

                    FWHM_z = d2;
                    FWHM_y = d2_normal;


//                    FWHM_z =FWHM_z*2;
                     current_distance = sqrtf(y_distance * y_distance/(FWHM_y*FWHM_y) + z_distance * z_distance/(FWHM_z*FWHM_z));
//                     if (current_distance <= FWHM)
//                        {
                            if (sum_vor[e] != 0.0f)  // in case MLEM
                            {
                            // with fast math
                            orthogonal_factor = 1 - (current_distance);
                            if (orthogonal_factor > 0.000f)
                                {

      //                        orthogonal_factor = 1 - (sqrtf(z_distance*z_distance+y_distance*y_distance) / sqrtf(d2 * d2 + d2_normal * d2_normal));
//                                adjust_coef_shared[i_s] += time_factor[e] * orthogonal_factor / sum_vor[e];
                                adjust_coef_shared[i_s] += time_factor[e] * orthogonal_factor / sum_vor[e];
                                }
                            }
//                        }
                 }

            }

        }

    }
    __syncthreads();
    adjust_coef[idt] = adjust_coef_shared[i_s];




}