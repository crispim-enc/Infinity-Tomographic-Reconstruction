#include <stdint.h>

//texture<char, 1> tex;

/*
Machine code for
STEP: FORWARD PROJECTION
CONFIGURATION: Multiple Kernel
PROJECTOR: BOX COUNTS
DOI CORRECTION: FALSE
DECAY CORRECTION: FALSE
*/

__global__ void forward_projection_cdrf
(const int n, const int m, const int p, const int start_x, const int start_y, const int start_z,
    int number_of_events, const int begin_event_gpu_limitation, const int end_event_gpu_limitation,
    float* aLeft, float* bLeft, float* cLeft, float* dLeft,
    float* aRight, float* bRight, float* cRight, float* dRight,
    float* aFront, float* bFront, float* cFront, float* dFront,
     float* aBack, float* bBack, float* cBack, float* dBack,
    float* sum_vor, const float* im_old, const float* system_matrix)
{
    const int shared_memory_size = 32;
    __shared__ float aLeftShared[shared_memory_size];
    __shared__ float bLeftShared[shared_memory_size];
    __shared__ float cLeftShared[shared_memory_size];
    __shared__ float dLeftShared[shared_memory_size];

    __shared__ float aRightShared[shared_memory_size];
    __shared__ float bRightShared[shared_memory_size];
    __shared__ float cRightShared[shared_memory_size];
    __shared__ float dRightShared[shared_memory_size];

    __shared__ float aFrontShared[shared_memory_size];
    __shared__ float bFrontShared[shared_memory_size];
    __shared__ float cFrontShared[shared_memory_size];
    __shared__ float dFrontShared[shared_memory_size];

    __shared__ float aBackShared[shared_memory_size];
    __shared__ float bBackShared[shared_memory_size];
    __shared__ float cBackShared[shared_memory_size];
    __shared__ float dBackShared[shared_memory_size];

    __shared__ float sum_vor_shared[shared_memory_size];
//    __shared__ char fov_cut_matrix_shared[shared_memory_size];

    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int e;
    float valueLeft;
    float valueRight;
    float valueFront;
    float valueBack;
    float probability;
    float term_diff_1;
    float term_diff_2;
    float term_sum_3;
    float term_sum_4;

//    float sum_vor_temp;
    int index;
    int x_t;
    int y_t;
    int z_t;
    const int number_events_max = end_event_gpu_limitation - begin_event_gpu_limitation;

    if (threadIdx.x >= shared_memory_size)
    {
        return;
    }
    if (threadId >= number_events_max)
    {
        return;
    }

    __syncthreads();
    e = threadId;
//    e= 0;
    int e_m = threadIdx.x;
    aLeftShared[e_m] = aLeft[e];
    bLeftShared[e_m] = bLeft[e];
    cLeftShared[e_m] = cLeft[e];
    dLeftShared[e_m] = dLeft[e];

    aRightShared[e_m] = aRight[e];
    bRightShared[e_m] = bRight[e];
    cRightShared[e_m] = cRight[e];
    dRightShared[e_m] = dRight[e];

    aFrontShared[e_m] = aFront[e];
    bFrontShared[e_m] = bFront[e];
    cFrontShared[e_m] = cFront[e];
    dFrontShared[e_m] = dFront[e];

    aBackShared[e_m] = aBack[e];
    bBackShared[e_m] = bBack[e];
    cBackShared[e_m] = cBack[e];
    dBackShared[e_m] = dBack[e];
    sum_vor_shared[e_m] = sum_vor[e];

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
                valueLeft = aLeftShared[e_m] * x_t + bLeftShared[e_m] * y_t + cLeftShared[e_m] * z_t - dLeftShared[e_m];

                if (valueLeft <= 0)
                {
                    valueRight = aRightShared[e_m] * x_t + bRightShared[e_m] * y_t + cRightShared[e_m] * z_t - dRightShared[e_m];
                    if (valueRight <= 0)
                    {
                        valueFront = aFrontShared[e_m] * x_t + bFrontShared[e_m] * y_t + cFrontShared[e_m] * z_t - dFrontShared[e_m];
                        if (valueFront <= 0)
                        {
                            valueBack = aBackShared[e_m] * x_t + bBackShared[e_m] * y_t + cBackShared[e_m] * z_t - dBackShared[e_m];
                            if (valueBack <= 0)
                            {
//                                 term_diff_1 = abs(valueLeft - valueRight)*0.5;
//                                term_diff_2 = abs(valueFront - valueBack)*0.5;
//                                term_sum_3 = abs(valueLeft + valueRight)*0.5;
//                                term_sum_4 = abs(valueFront + valueBack)*0.5;
////
//
//                                 probability = 1 -  sqrt(term_diff_1*term_diff_1 + term_diff_2*term_diff_2)/
//                                                        sqrt(term_sum_3*term_sum_3 + term_sum_4*term_sum_4);
//
                                 sum_vor_shared[e_m] +=  im_old[index];

                            }
                        }
                    }
                }
            }
        }
    }
    __syncthreads();
    sum_vor[e] = sum_vor_shared[e_m];

}