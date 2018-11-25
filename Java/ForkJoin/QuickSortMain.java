import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;
import java.util.concurrent.ForkJoinPool;

public class QuickSortMain {

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

    public static void main(String[] args) {
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

            QuickSortAction quickSort = new QuickSortAction(arr);
            ForkJoinPool pool = new ForkJoinPool();
            
            pool.invoke(quickSort);
            pool.shutdown();

            stopTime = System.currentTimeMillis();
            acum += (stopTime - startTime);
        }

        printArray(arr, n, "Sorted Array: ", row);
        System.out.printf("\nAverage Time Taken: %.4f ms\n", (acum / iter));

        keyboard.close();

    }
}