/* This program requires compute_35 architecture to call a global function recursively */
/* Compile using the following arguments: nvcc -arch=sm_35 -rdc=true -o quicksort_parallel -lcudadevrt quicksort_parallel.cu*/ 
#include <iostream>
#include <cstdio>
#include "header.h"

#define MAX_DEPTH 16
#define SELECTION_SORT 32

/* A utility function to fill an array of size n */
void fillArray(int* arr, int size)
{
    // Initializes random number generator
    time_t seed;
    srand((int) time(&seed));

    for (int i = 0; i < size; i++) {
        arr[i] = rand() % 51;
    }
}

/* A utility function to print array of size n */
void printArray(int* arr, int size, std::string banner, int row)
{
    std::cout << "\n" << banner << "\n" << std::endl;
    for (int i = 0; i < size; i++)
    {
        // Checks if i is divisible by the row size and makes a line jump
        // also checks that it's not 0 to prevent a line jump at the start
        if((i+1) % row == 0 && (i+1) > 0)
        {
            // Checks that the next locality is valid, otherwise it'll print
            // trash for locality size + 1
            if(i+1 < size){
                printf("%d\n", arr[i]);
                // Makes it so the last number of a row is not repeated on next line
                ++i;
            }
        }
        printf("%d | ", arr[i]);
    }
    printf("\n");
}

// Selection sort used when depth gets too big or the number of elements drops
// below a threshold.
__device__ void selectionSort(int* arr, int left, int right)
{
    for (int i = left; i <= right; ++i)
    {
        int min_val = arr[i];
        int min_idx = i;

        // Find the smallest value in the range [left, right].
        for (int j = i + 1; j <= right; ++j)
        {
            int val_j = arr[j];
            if (val_j < min_val)
            {
                min_idx = j;
                min_val = val_j;
            }
        }

        // Swap the values.
        if (i != min_idx)
        {
            arr[min_idx] = arr[i];
            arr[i] = min_val;
        }
    }
}

// Very basic quicksort algorithm, recursively launching the next level.
__global__ void quickSort(int* arr, int left, int right, int depth)
{
    // If we're too deep or there are few elements left, we use an selection sort
    if (depth >= MAX_DEPTH || right - left <= SELECTION_SORT)
    {
        selectionSort(arr, left, right);
        return;
    }

    int *lft_ptr = arr + left;
    int *rgt_ptr = arr + right;
    int pivot = arr[(left + right) / 2];

    // Do the partitioning.
    while (lft_ptr <= rgt_ptr)
    {
        // Find the next left-hand and right-hand values to swap
        int lft_val = *lft_ptr;
        int rgt_val = *rgt_ptr;

        // Move the left pointer as long as the pointed element is smaller than the pivot.
        while (lft_val < pivot)
        {
            lft_ptr++;
            lft_val = *lft_ptr;
        }

        // Move the right pointer as long as the pointed element is larger than the pivot.
        while (rgt_val > pivot)
        {
            rgt_ptr--;
            rgt_val = *rgt_ptr;
        }

        // If the swap points are valid swap them
        if (lft_ptr <= rgt_ptr)
        {
            *lft_ptr++ = rgt_val;
            *rgt_ptr-- = lft_val;
        }
    }

    // Now the recursive part
    int nright = rgt_ptr - arr;
    int nleft = lft_ptr - arr;

    // Launch a new block to sort the left part.
    if (left < (rgt_ptr - arr))
    {
        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        quickSort<<<1, 1, 0, s>>>(arr, left, nright, depth + 1);
        cudaStreamDestroy(s);
    }

    // Launch a new block to sort the right part.
    if ((lft_ptr - arr) < right)
    {
        cudaStream_t s1;
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        quickSort<<<1, 1, 0, s1>>>(arr, nleft, right, depth + 1);
        cudaStreamDestroy(s1);
    }
}

// Call the quicksort kernel from the host.
void runQuickSort(int* arr, int n)
{
    // Prepare CDP for the max depth 'MAX_DEPTH'.
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH);

    // Launch on device
    int left = 0;
    int right = n - 1;
    std::cout << "Launching kernel on the GPU" << std::endl;
    quickSort<<<1, 1>>>(arr, left, right, 0);
    cudaDeviceSynchronize();
}

// Main entry point.
int main(int argc, char **argv)
{
    int n = 0;
    int row = 0;
    int iter = 0;
    double acum = 0;

    while(1)
    {
        printf("Input an array length to sort:\n");
        scanf("%d", &n);

        if(n > 0)
            break;
    }

    while(1)
    {
        printf("Input number of iterations:\n");
        scanf("%d", &iter);

        if(iter > 0)
            break;
    }

    while(1)
    {
        printf("Input a row lenght:\n");
        scanf("%d", &row);

        if(row > 0)
            break;
    }

    // Get device properties
    int device_count = 0, device = -1;
    cudaGetDeviceCount(&device_count);

    for (int i = 0; i < device_count; ++i)
    {
        cudaDeviceProp properties;
        cudaGetDeviceProperties(&properties, i);
        if (properties.major > 3 || (properties.major == 3 && properties.minor >= 5))
        {
            device = i;
            std::cout << "Running on GPU " << i << " (" << properties.name << ")" << std::endl;
            break;
        }
        std::cout << "GPU " << i << " (" << properties.name << ") does not support CUDA Dynamic Parallelism" << std::endl;
    }

    if (device == -1)
    {
        std::cerr << "cdpSimpleQuicksort requires GPU devices with compute SM 3.5 or higher.  Exiting..." << std::endl;
        exit(EXIT_SUCCESS);
    }
    cudaSetDevice(device);

    // Create input array
    int *h_arr = 0;
    int *d_arr = 0;

    // Allocate CPU memory.
    h_arr = (int *)malloc(n * sizeof(int));

    // Allocate GPU memory.
    cudaMalloc((void **)&d_arr, n * sizeof(int));
    

    for (int i = 0; i < iter; i++) {
        fillArray(h_arr, n);
        cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);

        start_timer();
        // Execute
        std::cout << "Running quicksort on " << n << " elements" << std::endl;
        runQuickSort(d_arr, n);

		acum += stop_timer();
        // Copy result back
        cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    }

    // Print result
    printArray(h_arr, n, "Sorted Array: ", 20);
    // Print time taken
    printf("\nAverage Time taken: %.4lf ms\n", (acum / iter));

    free(h_arr);
    cudaFree(d_arr);
}
