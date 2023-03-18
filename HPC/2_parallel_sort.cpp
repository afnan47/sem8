    #include <iostream>
    #include <cstdlib>
    #include <ctime>
    #include <chrono>
    #include <omp.h>
     
    using namespace std;
     
    // Function to perform sequential bubble sort
    void sequential_bubble_sort(int arr[], int n) {
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                if (arr[j] > arr[j + 1]) {
                    int temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                }
            }
        }
    }
     
    // Function to perform parallel bubble sort
    void parallel_bubble_sort(int arr[], int n) {
        #pragma omp parallel
        {
            for (int i = 0; i < n - 1; i++) {
                int thread_num = omp_get_thread_num();
                int num_threads = omp_get_num_threads();
                for (int j = thread_num; j < n - i - 1; j += num_threads) {
                    if (arr[j] > arr[j + 1]) {
                        int temp = arr[j];
                        arr[j] = arr[j + 1];
                        arr[j + 1] = temp;
                    }
                }
            }
        }
    }
     
    // Function to merge two sorted arrays
    void merge(int arr[], int l, int m, int r) {
        int n1 = m - l + 1;
        int n2 = r - m;
        int L[n1], R[n2];
        for (int i = 0; i < n1; i++) {
            L[i] = arr[l + i];
        }
        for (int j = 0; j < n2; j++) {
            R[j] = arr[m + 1 + j];
        }
        int i = 0, j = 0, k = l;
        while (i < n1 && j < n2) {
            if (L[i] <= R[j]) {
                arr[k] = L[i];
                i++;
            }
            else {
                arr[k] = R[j];
                j++;
            }
            k++;
        }
        while (i < n1) {
            arr[k] = L[i];
            i++;
            k++;
        }
        while (j < n2) {
            arr[k] = R[j];
            j++;
            k++;
        }
    }
     
    // Function to perform sequential merge sort
    void sequential_merge_sort(int arr[], int l, int r) {
        if (l < r) {
            int m = l + (r - l) / 2;
            sequential_merge_sort(arr, l, m);
            sequential_merge_sort(arr, m + 1, r);
            merge(arr, l, m, r);
        }
    }
     
    // Function to perform parallel merge sort
    void parallel_merge_sort(int arr[], int l, int r,int threads) {
        if (l < r) {
            int m = l + (r - l) / 2;
            if(threads>1 && r-l+1>=threads*1000){
                #pragma omp parallel sections
            {
                #pragma omp section
                {
                    parallel_merge_sort(arr, l, m,threads/2);
                }
                #pragma omp section
                {
                    parallel_merge_sort(arr, m + 1, r,threads- threads/2);
                }
            }
            }
            else{
                sequential_merge_sort(arr, l, m);
                sequential_merge_sort(arr, m + 1, r);
                merge(arr, l, m, r);
            }
        }
    }
     
    // Function to test the performance of sequential and parallel bubble sort
    void test_bubble_sort_performance(int n) {
        int* arr = new int[n];
        int* arr_copy = new int[n];
     
        // Initialize the array with random values
        srand(time(NULL));
        for (int i = 0; i < n; i++) {
            arr[i] = rand();
            arr_copy[i] = arr[i];
        }
     
        // Measure the execution time of sequential bubble sort
        auto start_time = chrono::high_resolution_clock::now();
        sequential_bubble_sort(arr, n);
        auto end_time = chrono::high_resolution_clock::now();
        auto sequential_sort_time = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();
     
        // Measure the execution time of parallel bubble sort
        start_time = chrono::high_resolution_clock::now();
        parallel_bubble_sort(arr_copy, n);
        end_time = chrono::high_resolution_clock::now();
        auto parallel_sort_time = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();
     
        // Print the execution times of the sequential and parallel bubble sort
        cout << "Sequential bubble sort time: " << sequential_sort_time << " microseconds" << endl;
        cout << "Parallel bubble sort time: " << parallel_sort_time << " microseconds" << endl;
     
        // Free the memory allocated for the arrays
        delete[] arr;
        delete[] arr_copy;
    }
     
    void test_merge_sort_performance(int n)
    {
        int* arr = new int[n];
        int* arr_copy = new int[n];
     
        // Initialize array with random values
        srand(time(NULL));
        for (int i = 0; i < n; i++) {
            arr[i] = rand()%1000;
            arr_copy[i] = arr[i];
        }
     
        // Perform sequential merge sort and measure time
        double start = omp_get_wtime();
        sequential_merge_sort(arr, 0, n - 1);
        double end = omp_get_wtime();
     
        std::cout << "Sequential merge sort took " << end-start << " seconds\n";
     
        // Perform parallel merge sort and measure time
        int threads = omp_get_max_threads();
        start = omp_get_wtime();
        // int num_threads = omp_get_max_threads();
        parallel_merge_sort(arr_copy, 0, n - 1,threads);
        end = omp_get_wtime();
     
        std::cout << "Parallel merge sort took " << end - start << " seconds\n";
     
        delete[] arr;
    }
     
     
    // Example usage
    int main(){
        int n = 1000000;
        // test_bubble_sort_performance(n);
        test_merge_sort_performance(n);
        return 0;
    }