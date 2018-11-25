#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <cutil_inline.h>

#define MAX_THREADS	128 
#define N 100000

int*	r_values;
int*	d_values;

void fillArray(int* arr, int size)
{
    // Initializes random number generator
    time_t seed;
    srand((unsigned) time(&seed));

    for (int i = 0; i < size; i++) {
        arr[i] = rand() % 50;
    }
}

// Kernel function
__global__ static void quicksort(int* values) {
#define MAX_LEVELS	300

   int pivot, L, R;
   int idx =  threadIdx.x + blockIdx.x * blockDim.x;
   int start[MAX_LEVELS];
   int end[MAX_LEVELS];

   start[idx] = idx;
   end[idx] = N - 1;
   while (idx >= 0) {
       L = start[idx];
       R = end[idx];
       if (L < R) {
           pivot = values[L];
           while (L < R) {
               while (values[R] >= pivot && L < R)
                   R--;
               if(L < R)
                   values[L++] = values[R];
               while (values[L] < pivot && L < R)
                   L++;
               if (L < R)
                   values[R--] = values[L];
           }
           values[L] = pivot;
           start[idx + 1] = L + 1;
           end[idx + 1] = end[idx];
           end[idx++] = L;
           if (end[idx] - start[idx] > end[idx - 1] - start[idx - 1]) {
                           // swap start[idx] and start[idx-1]
                           int tmp = start[idx];
                           start[idx] = start[idx - 1];
                           start[idx - 1] = tmp;

                           // swap end[idx] and end[idx-1]
                           tmp = end[idx];
                           end[idx] = end[idx - 1];
                           end[idx - 1] = tmp;
                   }

       }
       else
           idx--;
   }
}

// program main
int main(int argc, char **argv) {
    unsigned int hTimer;
    size_t size = N * sizeof(int);
    
    // allocate host memory
    r_values = (int*)malloc(size);
    
   // allocate device memory
       cutilSafeCall( cudaMalloc((void**)&d_values, size) );

   // allocate threads per block
       const unsigned int cThreadsPerBlock = 128;

   for (int i = 0; i < 5; ++i) {
        // initialize data set
        Init(r_values, i);

        // copy data to device	
       cutilSafeCall( cudaMemcpy(d_values, r_values, size, cudaMemcpyHostToDevice) );

       printf("Beginning kernel execution...\n");

       cutilCheckError( cutCreateTimer(&hTimer) );
        cutilSafeCall( cudaThreadSynchronize() );
       cutilCheckError( cutResetTimer(hTimer) );
        cutilCheckError( cutStartTimer(hTimer) );
   
       // execute kernel
        quicksort <<< MAX_THREADS / cThreadsPerBlock, MAX_THREADS / cThreadsPerBlock, cThreadsPerBlock >>> (d_values);
        cutilCheckMsg( "Kernel execution failed..." );

        cutilSafeCall( cudaThreadSynchronize() );
        cutilCheckError( cutStopTimer(hTimer) );
        double gpuTime = cutGetTimerValue(hTimer);

        printf( "\nKernel execution completed in %f ms\n", gpuTime );
    
        // copy data back to host
       cutilSafeCall( cudaMemcpy(r_values, d_values, size, cudaMemcpyDeviceToHost) );
    
        // test print
        /*for (int i = 0; i < N; i++) {
            printf("%d ", r_values[i]);
        }
        printf("\n");
       */

       // test
               printf("\nTesting results...\n");
               for (int x = 0; x < N - 1; x++) {
                       if (r_values[x] > r_values[x + 1]) {
                               printf("Sorting failed.\n");
                               break;
                       }
                       else
                               if (x == N - 2)
                                       printf("SORTING SUCCESSFUL\n");
               }

   }
    
    // free memory
   cutilSafeCall( cudaFree(d_values) );
    free(r_values);
    
    cutilExit(argc, argv);
    cudaThreadExit();
}