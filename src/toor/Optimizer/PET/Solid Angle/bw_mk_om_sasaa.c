#include <stdint.h>
//static texture<float, 1> tex;

__global__ void backprojection_cdrf
(int dataset_number, int n, int m, int p,const int start_x, const int start_y, const int start_z, const float* crystal_pitch_XY, const float* crystal_pitch_Z,
    const float distance_between_array_pixel, int number_of_events, const int begin_event_gpu_limitation,
    const int end_event_gpu_limitation, const float* a, const float* a_normal, const float* a_cf, const float* b,
    const float* b_normal, const float* b_cf, const float* c, const float* c_normal, const float* c_cf, const float* d, const float* d_normal,
    const float* d_cf, const short* A, const short* B, const short* C, float* adjust_coef, float* sum_vor,
    char* fov_cut_matrix, short* x_min, short* x_max, short* y_min, short* y_max, short* z_min, short* z_max)
   {
    extern __shared__ float adjust_coef_shared[];
    int idt = blockIdx.x * blockDim.x + threadIdx.x;




//    float d2;
//    float d2_normal;
//    float d2_cf;
    float normal_value;
    float value;
    float value_cf;
    int i_s = threadIdx.x;
    float width;
    float height;
    float distance;
    float solid_angle;
    int x_ind;
    int y_ind;
    int z_ind;


    int index;
    int e;

//    fov_cut_matrix_shared[i_s] = fov_cut_matrix[idt];
//     if ( fov_cut_matrix[idt] == 0)
//     {
//     return;
//     }
    int im_z;
    int b_q;

    // convert idt (1D INDEX) INTO 3d INDEX into x, y  and z

//    z_ind = idt / (m*n);
//    b_q = idt - (m*n) * z_ind;
//    y_ind = b_q / n;
//    x_ind = idt - n * y_ind - m * n * z_ind;

    // convert idt (1D INDEX) INTO 3d INDEX into x, y  and idt order is given by  idt = z + y * p + x * p * m;
//    x_ind = idt / (m * p);
//    y_ind = (idt - x_ind * m * p) / p;
//    z_ind = idt - x_ind * m * p - y_ind * p;
//
    z_ind = idt % p;
    y_ind = ( idt / p ) % m;
    x_ind = idt / ( p * m );

//    x_ind = idt % p;
//    y_ind = (idt / p) % m;
//    z_ind = idt / (p * m);






//    index = z_ind + y_ind *p +x_ind*p*m;
//    printf("idt: %i; x_ind : %d; y_ind : %i; z_ind : %i", idt, x_ind, y_ind, z_ind);
//    printf("idt: %d; y_ind : %d", idt, y_ind);
//    printf("idt: %d; z_ind : %d", idt, z_ind);
    if (idt >= n * m * p)
    {
        return;
    }
//       if (z_ind > z_max)
//    {
//    return;
//    }
//       if (z_ind < z_min)
//    {
//    return;
//    }
    x_ind += start_x;
    y_ind += start_y;
    z_ind += start_z ;

    __syncthreads();

    adjust_coef_shared[i_s] = adjust_coef[idt];



    for (int e = begin_event_gpu_limitation; e < end_event_gpu_limitation; e++)
       {
//             e = index_events[ev];
//             e = ev;
//               if (x_ind >= x_min[e] && x_ind < x_max[e])
//                {
//
//                    if (y_ind >= y_min[e] && y_ind < y_max[e])
//                    {
//                        if (z_ind >= z_min[e] && z_ind < z_max[e])
//                        {
          if (x_ind < x_max[e])
                {
                    if (x_ind >= x_min[e])
                    {
                    if (y_ind < y_max[e])
                    {
                    if (y_ind >= y_min[e])
                    {
                    if (z_ind < z_max[e])
                    {
                    if (z_ind >= z_min[e])
                    {

                    value = a[e] * x_ind + b[e] * y_ind+ c[e] * z_ind - d[e];

                    normal_value =  a_normal[e] * x_ind + b_normal[e] * y_ind + c_normal[e] * z_ind- d_normal[e];
                    if (value*value + normal_value*normal_value <= 2)
                    {
 //                    normal_value=  a_normal_shared[s] * x_ind + b_normal_shared[s] * y_ind + c_normal_shared[s] * z_ind- d_normal_shared[s];



//
                        if (sum_vor[e] > 0)
                        {
                            adjust_coef_shared[i_s] += 1/sum_vor[e];
//                            adjust_coef[index] += solid_angle;
                        }
                      }
                }
                }
                }
                }
                }
                }

            }

    adjust_coef[idt] = adjust_coef_shared[i_s];
    __syncthreads();

}
