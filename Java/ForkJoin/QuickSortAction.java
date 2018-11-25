import java.util.concurrent.RecursiveAction;

public class QuickSortAction extends RecursiveAction {
    private int[] arr;
    private int left;
    private int right;

    public QuickSortAction(int[] arr) {
        this.arr = arr;
        left = 0;
        right = arr.length - 1;
    }

    public QuickSortAction(int[] arr, int left, int right) {
        this.arr = arr;
        this.left = left;
        this.right = right;
    }
    
    private void swap(int[] array, int i, int j) {
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }

    private int partition(int[] array, int low, int high) {
        int pivot = array[low];
        int i = low - 1;
        int j = high + 1;

        while (true) {
            do {
                i++;
            } while (array[i] < pivot);

            do {
                j--;
            } while (array[j] > pivot);
            
            if (i >= j)
                return j;

            swap(array, i, j);
        }
    }

    @Override
    protected void compute() {
        if (left < right) {
            int pivot = partition(arr, left, right);
            invokeAll(new QuickSortAction(arr, left, pivot), new QuickSortAction(arr, pivot + 1, right));
        }
    }
}