#include <stdint.h>

texture<char, 1> tex;


__global__ void forward_projection_cdrf
(const int n, const int m, const int p, const int start_x, const int start_y, const int start_z,
    float* crystal_pitch_XY, float* crystal_pitch_Z, const float distance_between_array_pixel,
    int number_of_events, const int begin_event_gpu_limitation, const int end_event_gpu_limitation,
    float* a, float* a_normal, float* a_cf, float* b, float* b_normal, float* b_cf, float* c, float* c_normal,
    float* c_cf, float* d, float* d_normal, float* d_cf, float* sum_vor, const char* fov_cut_matrix,
    const float* im_old, float* m_values, float* b_values, float* m_values_at, float* b_values_at,
    float* max_D, float* inflex_points_x, float* linear_attenuation_A,
    float* linear_attenuation_B)
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
    __shared__ float m_values_init_shared[shared_memory_size];
    __shared__ float m_values_end_shared[shared_memory_size];
    __shared__ float b_values_init_shared[shared_memory_size];
    __shared__ float b_values_end_shared[shared_memory_size];
    __shared__ float m_values_at_shared[shared_memory_size];
    __shared__ float b_values_at_shared[shared_memory_size];
    __shared__ float max_D_shared[shared_memory_size];
    __shared__ float inflex_points_x_init_shared[shared_memory_size];
    __shared__ float inflex_points_x_end_shared[shared_memory_size];
    __shared__ float linear_attenuation_A_shared[shared_memory_size];
    __shared__ float linear_attenuation_B_shared[shared_memory_size];

    /*
   const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
   const int threadId= blockId * (blockDim.x * blockDim.y * blockDim.z)+ (threadIdx.z * (blockDim.x * blockDim.y))+ (threadIdx.y * blockDim.x)+ threadIdx.x;
   */
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int e;

    float d2;
    float d2_normal;
    float d2_cf;
    float value;
    float value_normal;
    float value_cf;
    float sum_vor_temp;
    int index;
    int x_t;
    int y_t;
    int z_t;
    float width;
    float height;
    float distance;
    float distance_other;
    float distance_crystal;
    float distance_at;
    float idrf;



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

    __syncthreads();
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
    m_values_init_shared[e_m] = m_values[2 * e];
    m_values_end_shared[e_m] = m_values[2 * e + 1];
    b_values_init_shared[e_m] = b_values[2 * e];
    b_values_end_shared[e_m] = b_values[2 * e + 1];
    m_values_at_shared[e_m] = m_values_at[e];
    b_values_at_shared[e_m] = b_values_at[e];
    max_D_shared[e_m] = max_D[e];
    inflex_points_x_init_shared[e_m] = inflex_points_x[2 * e];
    inflex_points_x_end_shared[e_m] = inflex_points_x[2 * e + 1];
    linear_attenuation_A_shared[e_m] = linear_attenuation_A[e];
    linear_attenuation_B_shared[e_m] = linear_attenuation_B[e];

    d2_normal = crystal_pitch_XY_shared[e_m] * sqrt(a_normal_shared[e_m] * a_normal_shared[e_m] + b_normal_shared[e_m] * b_normal_shared[e_m] + c_normal_shared[e_m] * c_normal_shared[e_m]);
    d2 = crystal_pitch_Z_shared[e_m] * sqrt(a_shared[e_m] * a_shared[e_m] + b_shared[e_m] * b_shared[e_m] + c_shared[e_m] * c_shared[e_m]);
    d2_cf = distance_between_array_pixel * sqrt(a_cf_shared[e_m] * a_cf_shared[e_m] + b_cf_shared[e_m] * b_cf_shared[e_m] + c_cf_shared[e_m] * c_cf_shared[e_m]);
    max_distance_projector = sqrt(d2 * d2 + d2_normal * d2_normal + d2_cf * d2_cf);

    for (int l = 0; l < n; l++)
    {
        x_t = l + start_x;
        for (int j = 0; j < m; j++)
        {
            y_t = j + start_y;
            for (int k = 0; k < p; k++)
            {
                /*
                index = l+j*n+k*m*n;
                fov_cut_matrix_shared[k]=fov_cut_matrix[k+j*p+l*p*m];
                index = l+j*n+k*m*n;
                */
                index = k + j * p + l * p * m;

                if (fov_cut_matrix[index] != 0)
                {

                    z_t = k + start_z;
                    value_normal = a_normal_shared[e_m] * x_t + b_normal_shared[e_m] * y_t + c_normal_shared[e_m] * z_t - d_normal_shared[e_m];

                    if (value_normal < d2_normal && value_normal >= -d2_normal)
                    {
                        value = a_shared[e_m] * x_t + b_shared[e_m] * y_t + c_shared[e_m] * z_t - d_shared[e_m];

                        if (value < d2 && value >= -d2)
                        {
                            value_cf = a_cf_shared[e_m] * x_t + b_cf_shared[e_m] * y_t + c_cf_shared[e_m] * z_t - d_cf_shared[e_m];


                            if (value_cf >= -d2_cf && value_cf < d2_cf)
                            {
                                if (sqrt(value * value + value_normal * value_normal + value_cf * value_cf) <= max_distance_projector)
                                {
                                    width = 2 * (crystal_pitch_Z_shared[e_m] - abs(value));
                                    height = 2 * (crystal_pitch_XY_shared[e_m] - abs(value_normal));

                                    distance = d2_cf + abs(value_cf);
                                    distance_other = abs(d2_cf + value_cf);
                                    distance_at = m_values_at_shared[e_m] * value + b_values_at_shared[e_m];

                                    if (value <= inflex_points_x_init_shared[e_m])
                                    {
                                        distance_crystal = m_values_init_shared[e_m] * value + b_values_init_shared[e_m];
                                        distance_at = 0;
                                    }

                                    else if (value >= inflex_points_x_end_shared[e_m])
                                    {
                                        distance_crystal = m_values_end_shared[e_m] * value + b_values_end_shared[e_m];
                                    }

                                    else
                                    {
                                        distance_crystal = max_D_shared[e_m];
                                    }

                                    idrf = (1 - exp(-linear_attenuation_A_shared[e_m] * distance_crystal)) * exp(-linear_attenuation_A_shared[e_m] * distance_at);
                                    if (idrf < 0)
                                    {

                                        idrf = 0;
                                    }

                                    sum_vor_shared[e_m] += idrf * im_old[index];



                                }
                                /*
                                pow(4*atan(width*height/(2*distance*sqrt(4*distance*distance+width*width+height*height))),2)
                                sum_vor_shared[e_m]+= im_old[index]*(1-sqrt(value*value+value_normal*value_normal+value_cf*value_cf)/max_distance_projector);
                                4*arctan(
                                4*np.arctan(width*height/(2*distance_to_anh*np.sqrt(4*distance_between_array_pixel-value_cf)**2+width**2+height**2)))
                                *(1-sqrt(value*value+value_normal*value_normal+value_cf*value_cf)/max_distance_projector)
                                */
                            }


                        }




                    }
                }

            }
        }
    }

    /*
     sum_vor[e]= sum_vor_temp;
    sum_vor[e]= sum_vor_shared[e_m];


    sum_vor[e]= sum_vor_temp;
    */
    __syncthreads();
    sum_vor[e] = sum_vor_shared[e_m];
}