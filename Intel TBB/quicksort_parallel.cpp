#include <iostream>
#include <stdio.h>
#include <ctime>
#include <algorithm>
#include <functional>
#include "tbb/tick_count.h"
#include "tbb/task_group.h"

void fillArray(int arr[], int size)
{
    // Initializes random number generator
    time_t seed;
    srand((unsigned) time(&seed));

    for (int i = 0; i < size; i++) {
        arr[i] = rand() % 50;
    }
}

/* Function to print an array */
void printArray(int arr[], int size, std::string banner, int row)
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

void copyArray(int *source, int *destination, const long size)
{
    for (int i = 0; i < size; i++)
        destination[i] = source[i];
}

void quickSort(int *arr, const long n)
{
    long i = 0, j = n;
    int pivot = arr[n / 2]; 
    do
    {
        while (arr[i] < pivot)
            i++;
        while (arr[j] > pivot)
            j--;
        if (i <= j)
        {
            std::swap(arr[i], arr[j]);
            i++;
            j--;
        }
    } while (i <= j);
    
    if (n < 100)
    {   
        // limits the minimum recursion size
        if (j > 0)
            quickSort(arr, j);
        if (n > i)
            quickSort(arr + i, n - i);
        return;
    }

    tbb::task_group g;
    g.run([&] {if (j > 0) quickSort(arr, j); });
	g.run([&] {if (n > i) quickSort(arr + i, n - i); });
	g.wait();
}

int main()
{
    int MAX_NUMBER, n;
    std::cout << "Size of array: ";
    std::cin >> n;

    srand((int)time(0));
    int *arr = new int[n];

    fillArray(arr, n);
    printArray(arr, n, "Sorted Array: ", 20);

    tbb::tick_count start_par = tbb::tick_count::now();
    quickSort(arr, n - 1);
    tbb::tick_count end_par = tbb::tick_count::now();
    std::cout << "\nAverage Time Taken: " << (end_par - start_par).seconds() << std::endl;

    delete[] arr;

    return 0;
}