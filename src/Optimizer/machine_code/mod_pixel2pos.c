__global__ void pixeltoangle(const int n, const int m, const int p,  float* crystal_pitch_XY,  float* crystal_pitch_Z, const float distance_between_array_pixel, int number_of_events, const int begin_event_gpu_limitation, const int end_event_gpu_limitation, float* a, float* a_normal,
    float* a_cf, float* b, float* b_normal, float* b_cf, float* c, float* c_normal, float* c_cf, float* d, float* d_normal, float* d_cf, const int* A, const int* B,
    const int* C, float* sum_vor, float* im_old, const int* active_x, const int* active_y, const int* active_z, const int number_of_active_pixels)
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
    __shared__ float crystal_pitch_Z_shared[shared_memory_size];
    __shared__ float crystal_pitch_XY_shared[shared_memory_size];
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
    const int number_events_max = end_event_gpu_limitation - begin_event_gpu_limitation;
    const float error_pixel = 0.0000f;
    if (threadIdx.x > shared_memory_size)
    {
        return;
    }
    if (threadId >= number_events_max)
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
    crystal_pitch_XY_shared[e_m] = crystal_pitch_XY[e];
    crystal_pitch_Z_shared[e_m] = crystal_pitch_Z[e];
    __syncthreads();

    for (int l = 0; l < number_of_active_pixels; l++)
    {
        /*
        index = active_x[l]+active_y[l]*n+active_z[l]*n*m;
        */
        index = active_z[l] + active_y[l] * p + active_x[l] * p * m;

        value = a_shared[e_m] * A[index] + b_shared[e_m] * B[index] + c_shared[e_m] * C[index] - d_shared[e_m];
        d2 = crystal_pitch_XY_shared[e_m] * sqrt(a_shared[e_m] * a_shared[e_m] + b_shared[e_m] * b_shared[e_m] + c_shared[e_m] * c_shared[e_m]);
        if (value < d2 && value >= -d2)
        {
            value_normal = a_normal_shared[e_m] * A[index] + b_normal_shared[e_m] * B[index] + c_normal_shared[e_m] * C[index] - d_normal_shared[e_m];
            d2_normal =  crystal_pitch_Z_shared[e_m] * sqrt(a_normal_shared[e_m] * a_normal_shared[e_m] + b_normal_shared[e_m] * b_normal_shared[e_m] + c_normal_shared[e_m] * c_normal_shared[e_m]);


            if (value_normal < d2_normal && value_normal >= -d2_normal)

            {
//                value_cf = a_cf_shared[e_m] * A[index] + b_cf_shared[e_m] * B[index] + c_cf_shared[e_m] * C[index] - d_cf_shared[e_m];
//                d2_cf = sqrt(a_cf_shared[e_m] * a_cf_shared[e_m] + b_cf_shared[e_m] * b_cf_shared[e_m] + c_cf_shared[e_m] * c_cf_shared[e_m]);


                sum_vor_shared[e_m] += 1;
                im_old[index] += 1;


            }
        }



    }
    __syncthreads();
    sum_vor[e] = sum_vor_shared[e_m];

}