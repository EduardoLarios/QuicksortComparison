// C# program for implementation of QuickSort 
using System;
using System.Diagnostics;
using System.Threading;

namespace CFG
{
    class GFG
    {
        /* A utility function to fill an array of size n */
        static void fillArray(int[] arr, int size)
        {
            var generator = new Random();

            for (int i = 0; i < size; i++)
            {
                arr[i] = generator.Next(51);
            }
        }

        /* A utility function to print array of size n */
        static void printArray(int[] arr, int size, string banner, int row)
        {
            Console.WriteLine("\n{0}\n", banner);

            for (int i = 0; i < size; i++)
            {
                if ((i + 1) % row == 0 && (i + 1) > 0)
                {
                    if (i + 1 < size)
                    {
                        Console.WriteLine(arr[i]);
                        ++i;
                    }
                }
                Console.Write("{0} | ", arr[i]);
            }
            Console.WriteLine();
        }

        /* This function takes last element as pivot, 
        places the pivot element at its correct 
        position in sorted array, and places all 
        smaller (smaller than pivot) to left of 
        pivot and all greater elements to right 
        of pivot */
        static int partition(int[] arr, int low, int high)
        {
            int pivot = arr[high];

            // index of smaller element 
            int i = (low - 1);
            for (int j = low; j < high; j++)
            {
                // If current element is smaller 
                // than or equal to pivot 
                if (arr[j] <= pivot)
                {
                    i++;

                    // swap arr[i] and arr[j] 
                    int temp = arr[i];
                    arr[i] = arr[j];
                    arr[j] = temp;
                }
            }

            // swap arr[i+1] and arr[high] (or pivot) 
            int temp1 = arr[i + 1];
            arr[i + 1] = arr[high];
            arr[high] = temp1;

            return i + 1;
        }

        /* The main function that implements QuickSort() 
        arr[] --> Array to be sorted, 
        low --> Starting index, 
        high --> Ending index */
        static void quickSort(int[] arr, int low, int high)
        {
            if (low < high)
            {

                /* pi is partitioning index, arr[pi] is 
                now at right place */
                int pi = partition(arr, low, high);

                // Recursively sort elements before 
                // partition and after partition 
                quickSort(arr, low, pi - 1);
                quickSort(arr, pi + 1, high);
            }
        }

        // Driver program 
        public static void Main()
        {
            var stopwatch = new Stopwatch();
            int row, n, iter;

            while (true)
            {
                Console.WriteLine("Input an array length to sort: ");
                if (int.TryParse(Console.ReadLine(), out n))
                {
                    if (n > 0)
                        break;
                }

            }

            while (true)
            {
                Console.WriteLine("Input number of iterations: ");
                if (int.TryParse(Console.ReadLine(), out iter))
                {
                    if (iter > 0)
                        break;
                }
            }

            while (true)
            {
                Console.WriteLine("Input a row lenght: ");
                if (int.TryParse(Console.ReadLine(), out row))
                {
                    if (row > 0)
                        break;
                }
            }

            var arr = new int[n];

            for (int i = 0; i < iter; i++)
            {
                fillArray(arr, n);
                stopwatch.Start();
                quickSort(arr, 0, n - 1);
                stopwatch.Stop();
            }

            var ts = stopwatch.Elapsed;
            printArray(arr, n, "Sorted Array: ", row);
            Console.WriteLine("\nAverage Time Taken: {0:00}:{1:0000} milliseconds", ts.Seconds / iter, ts.Milliseconds / iter);
        }
    }
}