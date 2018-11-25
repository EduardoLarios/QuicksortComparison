#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include "header.h"

#define MAX_THREADS 128

/* A utility function to fill an array of size n */
void fillArray(int* arr, int size)
{
    // Initializes random number generator
    time_t seed;
    srand((unsigned) time(&seed));

    for (int i = 0; i < size; i++) {
        arr[i] = rand() % 50;
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

// Kernel function to sort array on GPU
__global__ static void quickSort(int* arr, int n)
{
    #define MAX_LEVELS 300

    int pivot, left, right;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int start[MAX_LEVELS];
    int end[MAX_LEVELS];

    start[idx] = idx;
    end[idx] = n - 1;
    while (idx >= 0)
    {
        left = start[idx];
        right = end[idx];

        if (left < right)
        {
            pivot = arr[left];
            while (left < right)
            {
                while (arr[right] >= pivot && left < right)
                {
                    right--;
                }

                if (left < right)
                    arr[left++] = arr[right];

                while (arr[left] < pivot && left < right)
                {
                    left++;
                }

                if (left < right)
                    arr[right--] = arr[left];
            }

            arr[left] = pivot;
            start[idx + 1] = left + 1;
            end[idx + 1] = end[idx];
            end[idx++] = left;

            if (end[idx] - start[idx] > end[idx - 1] - start[idx - 1])
            {
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
        {
            idx--;
        }
    }
}

// Driver program
int main()
{
    int n = 0;
    int row = 0;
    int iter = 0;

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

    int *arr; 
    int *dev_arr;
    int size = sizeof(int*) * n;
    
    arr = (int*) malloc(size);
    cudaMalloc((void**)&dev_arr, size);
    fillArray(arr, n);

    const unsigned int THREADS_PER_BLOCK = 128;
    double acum = 0;

	for (int i = 0; i < iter; i++) {
        cudaMemcpy(dev_arr, arr, size, cudaMemcpyHostToDevice);

        start_timer();
        quickSort<<<MAX_THREADS / THREADS_PER_BLOCK, MAX_THREADS / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(dev_arr, n);
        cudaThreadSynchronize();

        cudaMemcpy(arr, dev_arr, size, cudaMemcpyDeviceToHost);
		acum += stop_timer();
    }
    
    printArray(arr, n, "Sorted array: ", row);
    printf("\nAverage Time taken: %.4lf ms\n", (acum / iter));
    
    cudaFree(dev_arr);
    free(arr);

}
