#include <stdint.h>

texture<uint8_t, 1> tex;
__device__ float* three_plane_intersection(float plane1A, float plane1B,
    float plane1C, float plane1D, float plane2A, float plane2B, float plane2C,
    float plane2D, float plane3A, float plane3B, float plane3C, float plane3D);
__device__ float intersection_determinant(float matrix[3][3]);
__device__ float point_distance_to_plane(float* point, float A, float B, float C, float D);

__global__ void normalization_cdrf
(int dataset_number, int n, int m, int p, float* crystal_pitch_XY, float* crystal_pitch_Z,
    const float distance_between_array_pixel, int number_of_events, const int begin_event_gpu_limitation,
    const int end_event_gpu_limitation, const float* a, const float* a_normal, const float* a_cf, const float* b,
    const float* b_normal, const float* b_cf, const float* c, const float* c_normal, const float* c_cf, const float* d, const float* d_normal,
    const float* d_cf, const short* A, const short* B, const short* C, float* adjust_coef, float* sum_vor,
    char* fov_cut_matrix, float* time_factor, float* plane_centerA1_A,
    float* plane_centerA1_B, float* plane_centerA1_C, float* plane_centerA1_D,
    float* plane_centerB1_A, float* plane_centerB1_B, float* plane_centerB1_C,
    float* plane_centerB1_D, float* plane_centerC1_A, float* plane_centerC1_B,
    float* plane_centerC1_C, float* plane_centerC1_D, float* intersection_points, float* m_values, float* m_values_at,
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
    float solid_angle;
    float face_1_distance_to_center;
    float face_2_distance_to_center;
    float face_3_distance_to_center;
    float* p1;
    float* p2;
    float* p3;
    float* p4;
    float* p5;
    float* p6;
    float dist_p1;
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

                        width = 2 * (crystal_pitch_Z[e] - abs(value));
                        height = 2 * (crystal_pitch_XY[e] - abs(normal_value));
                        distance = d2_cf + abs(value_cf);

                        /*
                         face_1_distance_to_center = 1.0f * sqrt(plane_centerA1_A[e] * plane_centerA1_A[e] + plane_centerA1_B[e] * plane_centerA1_B[e] + plane_centerA1_C[e] * plane_centerA1_C[e]);
                         face_2_distance_to_center = 1.0f * sqrt(plane_centerB1_A[e] * plane_centerB1_A[e] + plane_centerB1_B[e] * plane_centerB1_B[e] + plane_centerB1_C[e] * plane_centerB1_C[e]);
                         face_3_distance_to_center = 15.0f * sqrt(plane_centerC1_A[e] * plane_centerC1_A[e] + plane_centerC1_B[e] * plane_centerC1_B[e] + plane_centerC1_C[e] * plane_centerC1_C[e]);

                         p1 = three_plane_intersection(a[e],
                             b[e], c[e], d[e],
                             a_normal[e], b_normal[e], c_normal[e],
                             d_normal[e], plane_centerA1_A[e], plane_centerA1_B[e],
                             plane_centerA1_C[e], plane_centerA1_D[e]+face_1_distance_to_center);

                         p2 = three_plane_intersection(a[e],
                             b[e], c[e], d[e],
                             a_normal[e], b_normal[e], c_normal[e],
                             d_normal[e], plane_centerA1_A[e], plane_centerA1_B[e],
                             plane_centerA1_C[e], plane_centerA1_D[e]-face_1_distance_to_center);

                         p3 = three_plane_intersection(a[e],
                             b[e], c[e], d[e],
                             a_normal[e], b_normal[e], c_normal[e],
                             d_normal[e], plane_centerB1_A[e], plane_centerB1_B[e],
                             plane_centerB1_C[e], plane_centerB1_D[e]+face_2_distance_to_center);

                         p4 = three_plane_intersection(a[e],
                             b[e], c[e], d[e],
                             a_normal[e], b_normal[e], c_normal[e],
                             d_normal[e], plane_centerB1_A[e], plane_centerB1_B[e],
                             plane_centerB1_C[e], plane_centerB1_D[e]-face_2_distance_to_center);

                         p5 = three_plane_intersection(a[e],
                             b[e], c[e], d[e],
                             a_normal[e], b_normal[e], c_normal[e],
                             d_normal[e], plane_centerC1_A[e], plane_centerC1_B[e],
                             plane_centerC1_C[e], plane_centerC1_D[e]+face_3_distance_to_center);

                         p6 = three_plane_intersection(a[e],
                             b[e], c[e], d[e],
                             a_normal[e], b_normal[e], c_normal[e],
                             d_normal[e], plane_centerC1_A[e], plane_centerC1_B[e],
                             plane_centerC1_C[e], plane_centerC1_D[e]-face_3_distance_to_center);

                         intersection_points[e*6] = point_distance_to_plane(p1, a_cf[e], b_cf[e],c_cf[e], d_cf[e]);
                         dist_p2 = point_distance_to_plane(p2, a_cf[e], b_cf[e],c_cf[e], d_cf[e]);



                          printf(" Distance p1: %f", intersection_points[e*6]);

                           dist_p2 = point_distance_to_plane(p2, a_cf[e], b_cf[e],c_cf[e], d_cf[e]);
                         dist_p3 = point_distance_to_plane(p3, a_cf[e], b_cf[e],c_cf[e], d_cf[e]);
                         dist_p4 = point_distance_to_plane(p4, a_cf[e], b_cf[e],c_cf[e], d_cf[e]);
                         dist_p5 = point_distance_to_plane(p5, a_cf[e], b_cf[e],c_cf[e], d_cf[e]);
                         dist_p6 = point_distance_to_plane(p6, a_cf[e], b_cf[e],c_cf[e], d_cf[e]);
                         printf(" Distance p1: %f", dist_p1);
                         printf(" x: %f, y: %f, z: %f ",p1[0], p1[1], p1[2]);
                         printf("----------------");
                         printf(" x_2: %f, y_2: %f, z_2: %f ",p2[0], p2[1], p2[2]);
                       */
                        distance_at = m_values_at[e] * value + b_values_at[e];

                        if (value <= inflex_points_x[2 * e])
                        {
                            distance_crystal = m_values[2 * e] * value + b_values[2 * e];
                            distance_at = 0;
                        }
                        else if (value >= inflex_points_x[2 * e + 1])
                        {
                            distance_crystal = m_values[2 * e + 1] * value + b_values[2 * e + 1];
                        }

                        else
                        {
                            distance_crystal = max_D[e];


                        }
                        distance_crystal = max_D[e];
                        idrf = (1 - exp(-linear_attenuation_A[e] * distance_crystal)) * exp(-linear_attenuation_A[e] * distance_at);



                        solid_angle = 4 * width * height / (2 * distance * sqrt(4 * distance * distance + width * width + height * height));


                        adjust_coef_shared[i_s] += idrf / (sum_vor[e]);

                        /*
                     else if(value<=inflex_points_x[2*e+1])
                        {
                            distance_crystal = m_values[2*e+1]*value+b_values[2*e+1];
                        }

                           printf(" x_2: %f,  distance: %f, att: %f ",idrf,distance_crystal, linear_attenuation_A[e] );
                         if (sqrt(value*value+normal_value*normal_value+value_cf*value_cf)<=sqrt(d2*d2+d2_normal*d2_normal+d2_cf*d2_cf))
                        {
                           adjust_coef_shared[i_s] += time_factor[e]*(1-(sqrt(value*value+normal_value*normal_value+value_cf*value_cf))/sqrt(d2*d2+d2_normal*d2_normal+d2_cf*d2_cf))/(sum_vor[e]);
                            }
                          */


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
__device__ float point_distance_to_plane(float* point, float A, float B, float C, float D)
{
    float distance;
    distance = abs(A * point[0] + B * point[1] + C * point[2] - D) / sqrt(A * A + B * B + C * C);
    return distance;
}

__device__  float* three_plane_intersection(float plane1A, float plane1B,
    float plane1C, float plane1D, float plane2A, float plane2B, float plane2C,
    float plane2D, float plane3A, float plane3B, float plane3C, float plane3D)
{
    float m_det[3][3];
    float m_x[3][3];
    float m_y[3][3];
    float m_z[3][3];
    float det;
    float det_x;
    float det_y;
    float det_z;
    float result[3];

    m_det[0][0] = plane1A;
    m_det[0][1] = plane1B;
    m_det[0][2] = plane1C;
    m_det[1][0] = plane2A;
    m_det[1][1] = plane2B;
    m_det[1][2] = plane2C;
    m_det[2][0] = plane3A;
    m_det[2][1] = plane3B;
    m_det[2][2] = plane3C;

    m_x[0][0] = plane1D;
    m_x[0][1] = plane1B;
    m_x[0][2] = plane1C;
    m_x[1][0] = plane2D;
    m_x[1][1] = plane2B;
    m_x[1][2] = plane2C;
    m_x[2][0] = plane3D;
    m_x[2][1] = plane3B;
    m_x[2][2] = plane3C;

    m_y[0][0] = plane1A;
    m_y[0][1] = plane1D;
    m_y[0][2] = plane1C;
    m_y[1][0] = plane2A;
    m_y[1][1] = plane2D;
    m_y[1][2] = plane2C;
    m_y[2][0] = plane3A;
    m_y[2][1] = plane3D;
    m_y[2][2] = plane3C;

    m_z[0][0] = plane1A;
    m_z[0][1] = plane1B;
    m_z[0][2] = plane1D;
    m_z[1][0] = plane2A;
    m_z[1][1] = plane2B;
    m_z[1][2] = plane2D;
    m_z[2][0] = plane3A;
    m_z[2][1] = plane3B;
    m_z[2][2] = plane3D;

    det = intersection_determinant(m_det);
    det_x = intersection_determinant(m_x);
    det_y = intersection_determinant(m_y);
    det_z = intersection_determinant(m_z);


    if (det != 0.0f)
    {
        result[0] = det_x / det;
        result[1] = det_y / det;
        result[2] = det_z / det;
    }

    return result;
}

__device__ float intersection_determinant(float matrix[3][3])
{
    float a;
    float b;
    float c;
    float d;
    float e;
    float f;
    float g;
    float h;
    float i;
    float det;

    a = matrix[0][0];
    b = matrix[0][1];
    c = matrix[0][2];
    d = matrix[1][0];
    e = matrix[1][1];
    f = matrix[1][2];
    g = matrix[2][0];
    h = matrix[2][1];
    i = matrix[2][2];

    det = (a * e * i + b * f * g + c * d * h) - (a * f * h + b * d * i + c * e * g);
    return det;
}