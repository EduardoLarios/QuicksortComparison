import java.util.Scanner;
import java.util.Random;

// Java program for the implementation of QuickSort 
class QuickSort {

    /* A utility function to fill an array of size n */
    static void fillArray(int arr[], int size) {
        var generator = new Random();

        for (int i = 0; i < size; i++) {
            arr[i] = generator.nextInt(51);
        }
    }

    /* A utility function to print array of size n */
    static void printArray(int arr[], int size, String banner, int row) {
        System.out.println("\n" + banner + "\n");

        for (int i = 0; i < size; i++) {
            if ((i + 1) % row == 0 && (i + 1) > 0) {
                if (i + 1 < size) {
                    System.out.println(arr[i]);
                    ++i;
                }
            }
            System.out.print(arr[i] + " | ");
        }
        System.out.println();
    }

    /*
     * This function takes the last element as pivot, places the pivot element at
     * its correct position in sorted array, and places all smaller (smaller than
     * pivot) to left of pivot and all greater elements to right of pivot
     */
    static int partition(int arr[], int low, int end) {
        int pivot = arr[end];
        int i = (low - 1); // index of smaller element
        for (int j = low; j < end; j++) {
            // If current element is smaller than or
            // equal to pivot
            if (arr[j] <= pivot) {
                i++;

                // swap arr[i] and arr[j]
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }

        // swap arr[i+1] and arr[end] (or pivot)
        int temp = arr[i + 1];
        arr[i + 1] = arr[end];
        arr[end] = temp;

        return i + 1;
    }

    /*
     * The main function that implements QuickSort() arr[] --> Array to be sorted,
     * low --> Starting index, end --> Ending index
     */
    static void sortArray(int arr[], int start, int end) {
        if (start < end) {
            // pi is partitioning index, arr[pi] is now at right place
            int pi = partition(arr, start, end);

            // Recursively sort elements before
            // partition and after partition
            sortArray(arr, start, pi - 1);
            sortArray(arr, pi + 1, end);
        }
    }

    // Driver program to run the sort
    public static void main(String args[]) {
        var keyboard = new Scanner(System.in);

        long startTime, stopTime;
        int row = 0;
        int n = 0;
        int iter = 0;

        while (true) {
            System.out.println("Input an array length to sort: ");
            n = keyboard.nextInt();

            if (n > 0)
                break;
        }

        while (true) {
            System.out.println("Input number of iterations: ");
            iter = keyboard.nextInt();

            if (iter > 0)
                break;
        }

        while (true) {
            System.out.println("Input a row lenght: ");
            row = keyboard.nextInt();

            if (row > 0)
                break;
        }

        var arr = new int[n];
        double acum = 0;

        for (int i = 0; i < iter; i++) {
            fillArray(arr, n);

            startTime = System.currentTimeMillis();
            sortArray(arr, 0, n - 1);
            stopTime = System.currentTimeMillis();
            acum += (stopTime - startTime);
        }

        printArray(arr, n, "Sorted Array: ", row);
        System.out.printf("\nAverage Time Taken: %.4f ms\n", (acum / iter));

        keyboard.close();
    }
}
