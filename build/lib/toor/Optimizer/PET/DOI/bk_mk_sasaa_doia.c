#include <stdint.h>
texture<uint8_t, 1> tex;

__global__ void backprojection_cdrf
(int dataset_number, int n, int m, int p, const float* crystal_pitch_XY, const float* crystal_pitch_Z,
    const float distance_between_array_pixel, int number_of_events, const int begin_event_gpu_limitation,
    const int end_event_gpu_limitation, const float* a, const float* a_normal, const float* a_cf, const float* b,
    const float* b_normal, const float* b_cf, const float* c, const float* c_normal, const float* c_cf, const float* d, const float* d_normal,
    const float* d_cf, const short* A, const short* B, const short* C, float* adjust_coef, float* sum_vor,
    char* fov_cut_matrix, float* time_factor, float* m_values, float* m_values_at,
    float* b_values, float* b_values_at, float* max_D, float* inflex_points_x, float* linear_attenuation_A,
    float* linear_attenuation_B)
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
    float distance_crystal;
    float distance_at;
    float idrf;

    if (idt >= n * m * p)
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
            d2_normal = crystal_pitch_XY[e] * sqrt(a_normal[e] * a_normal[e] + b_normal[e] * b_normal[e] + c_normal[e] * c_normal[e]);

            if (normal_value < d2_normal && normal_value >= -d2_normal)
            {
                value = a[e] * a_temp + b[e] * b_temp + c[e] * c_temp - d[e];
                d2 = crystal_pitch_Z[e] * sqrt(a[e] * a[e] + b[e] * b[e] + c[e] * c[e]);


                if (value < d2 && value >= -d2)
                {
                    value_cf = a_cf[e] * a_temp + b_cf[e] * b_temp + c_cf[e] * c_temp - d_cf[e];
                    d2_cf = distance_between_array_pixel * sqrt(a_cf[e] * a_cf[e] + b_cf[e] * b_cf[e] + c_cf[e] * c_cf[e]);


                    if (value_cf >= -d2_cf && value_cf < d2_cf)
                    {
                        if (sqrt(value * value + normal_value * normal_value + value_cf * value_cf) <= sqrt(d2 * d2 + d2_normal * d2_normal + d2_cf * d2_cf))
                        {
                            width = 2 * (crystal_pitch_Z[e] - abs(value));
                            height = 2 * (crystal_pitch_XY[e] - abs(normal_value));
                            distance = d2_cf + abs(value_cf);
                            distance_other = d2_cf - abs(value_cf);
                            distance_at = m_values_at[e] * value + b_values_at[e];

                            if (value <= inflex_points_x[2 * e])
                            {
                                distance_crystal = m_values[2 * e] * value + b_values[2 * e];
                                distance_at = 0;
                            }
                            else if (value > inflex_points_x[2 * e + 1])
                            {
                                distance_crystal = m_values[2 * e + 1] * value + b_values[2 * e + 1];
                            }

                            else
                            {
                                distance_crystal = max_D[e];


                            }

                            idrf = (1 - exp(-linear_attenuation_A[e] * distance_crystal)) * exp(-linear_attenuation_A[e] * distance_at);

                            if (idrf < 0)
                            {
                                idrf = 0;
                            }
                            solid_angle = 4 * width * height / (2 * distance * sqrt(4 * distance * distance + width * width + height * height));

                            if (sum_vor[e] != 0)
                            {
                                adjust_coef_shared[i_s] += idrf / sum_vor[e];
                            }


                            /*
                            (4 * asin(sin(tan(width/distance))*sin(tan(height/distance))))*(4 * asin(sin(tan(width/distance))*sin(tan(height/distance))))/sum_vor[e];
                        adjust_coef_shared[i_s] += time_factor[e]*(1-(sqrt(value*value+normal_value*normal_value+value_cf*value_cf))/sqrt(d2*d2+d2_normal*d2_normal+d2_cf*d2_cf))/(sum_vor[e]);

                            */
                        }

                    }

                    /*
                    (1-(sqrt(value*value+normal_value*normal_value+value_cf*value_cf))/sqrt(d2*d2+d2_normal*d2_normal+d2_cf*d2_cf))/
                    normal_value= a_normal[e]*a_temp+b_normal[e]*b_temp+c_normal[e] * c_temp-d_normal[e];
     d2_normal = error_pixel+crystal_pitch_XY * sqrt(a_normal[e]*a_normal[e]+b_normal[e]*b_normal[e]+c_normal[e]*c_normal[e]);
                   adjust_coef_shared[i_s] += /(sum_vor[e]*time_factor[e]);
                  adjust_coef_shared[i_s] += (1-(sqrt(value*value+normal_value*normal_value+value_cf*value_cf))/sqrt(d2*d2+d2_normal*d2_normal+d2_cf*d2_cf))/(sum_vor[0]*time_factor[e]);
                     */





                }
            }

        }

    }

    adjust_coef[idt] = adjust_coef_shared[i_s];
    __syncthreads();
}