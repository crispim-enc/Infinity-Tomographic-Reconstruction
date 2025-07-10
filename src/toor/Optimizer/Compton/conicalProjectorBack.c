#include <stdint.h>
// Not complete
__global__ void backprojection_cdrf
(int dataset_number, int n, int m, int p, int number_of_events, const int begin_event_gpu_limitation,
    const int end_event_gpu_limitation,
    const float* aLeft, const float* bLeft, const float* cLeft, const float* dLeft,
    const float* aRight, const float* bRight, const float* cRight, const float* dRight,
     const float* aFront, const float* bFront, const float* cFront, const float* dFront,
     const float* aBack, const float* bBack,   const float* cBack, const float* dBack,
     const short* A, const short* B, const short* C,
    float* adjust_coef, float* sum_vor, float* system_matrix, float* im_gpu, int* counts)
{

    extern __shared__ float adjust_coef_shared[];
    int idt = blockIdx.x * blockDim.x + threadIdx.x;

    float valueLeft;
    float valueRight;
    float valueFront;
    float valueBack;
    float term_diff_1;
    float term_diff_2;
    float term_sum_3;
    float term_sum_4;

    short a_temp;
    short b_temp;
    short c_temp;
    float system_matrix_temp;
    float im_gpu_temp;
//    char fov_cut_temp;
    int i_s = threadIdx.x;
    float probability;


    if (idt > n * m * p)
    {
        return;
    }

    __syncthreads();
    adjust_coef_shared[i_s] = adjust_coef[idt];
    system_matrix_temp = system_matrix[idt];
    im_gpu_temp =  im_gpu[idt];
    a_temp = A[idt];
    b_temp = B[idt];
    c_temp = C[idt];


    for (int e = begin_event_gpu_limitation; e < end_event_gpu_limitation; e++)
    {
        valueLeft = aLeft[e] * a_temp + bLeft[e] * b_temp + cLeft[e] * c_temp - dLeft[e];

        if (valueLeft <= 0)
        {
            valueRight = aRight[e] * a_temp + bRight[e] * b_temp + cRight[e] * c_temp - dRight[e];
            if (valueRight <= 0)
            {
                valueFront = aFront[e] * a_temp + bFront[e] * b_temp + cFront[e] * c_temp - dFront[e];
                if (valueFront <= 0)
                {
                    valueBack = aBack[e] * a_temp + bBack[e] * b_temp + cBack[e] * c_temp - dBack[e];
                    if (valueBack <= 0)
                    {
                         if (sum_vor[e] != 0)
                            {

//                                term_diff_1 = abs(valueLeft - valueRight)*0.5;
//                                term_diff_2 = abs(valueFront - valueBack)*0.5;
//                                term_sum_3 = abs(valueLeft + valueRight)*0.5;
//                                term_sum_4 = abs(valueFront + valueBack)*0.5;
////                                probability = 1 - sqrt(((abs(valueLeft) - abs(valueRight))*0.5) +
////                                                        ((abs(valueFront) - abs(valueBack))*0.5)**2)/
////                                                        sqrt(((abs(valueLeft) + abs(valueRight))*0.5)**2 +
////                                                        ((abs(valueFront) + abs(valueBack))*0.5)**2);
//
//                                 probability = 1 -  sqrt(term_diff_1*term_diff_1 + term_diff_2*term_diff_2)/
//                                                        sqrt(term_sum_3*term_sum_3 + term_sum_4*term_sum_4);

//                                adjust_coef_shared[i_s] = 1* expf(-sum_vor[e])*(1 - expf(1 ));
                                adjust_coef_shared[i_s]+=  counts[e]/sum_vor[e];
//                                adjust_coef_shared[i_s] += 1;
                            }
                    }
                }
            }
        }
    }

    adjust_coef[idt] = adjust_coef_shared[i_s];
    __syncthreads();

}