/* C implementation QuickSort */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "header.h"

// A utility function to swap two elements
void swap(int *a, int *b)
{
    int t = *a;
    *a = *b;
    *b = t;
}

void fillArray(int arr[], int size)
{
    // Initializes random number generator
    time_t seed;
    srand((unsigned)time(&seed));

    for (int i = 0; i < size; i++)
    {
        arr[i] = rand() % 50;
    }
}

/* Function to print an array */
void printArray(int arr[], int size, char banner[], int row)
{
    printf("\n%s\n", banner);
    for (int i = 0; i < size; i++)
    {
        // Checks if i is divisible by the row size and makes a line jump
        // also checks that it's not 0 to prevent a line jump at the start
        if ((i + 1) % row == 0 && (i + 1) > 0)
        {
            // Checks that the next locality is valid, otherwise it'll print
            // trash for locality size + 1
            if (i + 1 < size)
            {
                printf("%d\n", arr[i]);
                // Makes it so the last number of a row is not repeated on next line
                ++i;
            }
        }
        printf("%d | ", arr[i]);
    }
    printf("\n");
}

/* 
    This function takes the last element as pivot, places 
    the pivot element at its correct position in the sorted 
	array, and places all smaller elements to the left of the
    pivot and all greater elements to its right 
*/
int partition(int arr[], int low, int high)
{
    int pivot = arr[high]; // pivot
    int i = (low - 1);     // Index of smaller element

    for (int j = low; j <= high - 1; j++)
    {
        // If current element is smaller than or
        // equal to pivot
        if (arr[j] <= pivot)
        {
            i++; // increment index of smaller element
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

/* 
    The main function that implements QuickSort 
    arr[] --> Array to be sorted, 
    low --> Starting index, 
    high --> Ending index 
*/
void quickSort(int arr[], int low, int high)
{
    if (low < high)
    {
        // pi is the partitioning index, arr[p] is now
        // at the right place
        int pi = partition(arr, low, high);

        // Separately sort elements before
        // partition and after partition
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

// Driver program to test above functions
int main()
{
    int n = 0;
    int row = 0;
    int iter = 0;

    while (1)
    {
        printf("Input an array length to sort:\n");
        scanf("%d", &n);

        if (n > 0)
            break;
    }

    while (1)
    {
        printf("Input number of iterations:\n");
        scanf("%d", &iter);

        if (iter > 0)
            break;
    }
    
    while (1)
    {
        printf("Input a row lenght:\n");
        scanf("%d", &row);

        if (row > 0)
            break;
    }


    int arr[n];
    double acum = 0;

    for (int i = 0; i < iter; i++)
    {
        fillArray(arr, n);

        start_timer();
        quickSort(arr, 0, n - 1);
        acum += stop_timer();
    }

    printArray(arr, n, "Sorted array: ", row);
    printf("\nAverage Time taken: %.4lf ms\n", (acum / iter));

    return 0;
}
