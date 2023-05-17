#include<iostream>
#include<omp.h>
using namespace std;


void bubble(int array[], int n){
    for (int i = 0; i < n - 1; i++){
        for (int j = 0; j < n - i - 1; j++){
            if (array[j] > array[j + 1]) swap(array[j], array[j + 1]);
        }
    }
}

void pBubble(int array[], int n){
    int i,j;
    #pragma omp parallel for shared(array, n) private(i, j)
    for (i = 0; i < n - 1; i++) {
        for (j = 0; j < n - i - 1; j++) {
            if (array[j] > array[j + 1]) swap(array[j], array[j + 1]);
        }
    }
}

int main(){
    int n = 10000;
    int a[n];
    double start_time, end_time;

    // Create an array of n numbers, with numbers from n to 1
    for(int i = 0, j = n; i < n; i++, j--) a[i] = j;
    
    // Create a copy 
    int b[n];
    for(int i = 0; i < n; i++) b[i] = a[i];

    // Measure Sequential Time
    start_time = omp_get_wtime(); 
    bubble(a,n);
    end_time = omp_get_wtime(); 
    cout << "Time taken by sequential algorithm: " << end_time - start_time << " seconds\n";

    //Measure Parallel time
    start_time = omp_get_wtime(); 
    pBubble(b,n);
    end_time = omp_get_wtime(); 
    cout << "Time taken by parallel algorithm: " << end_time - start_time << " seconds";
    
    /*
    The parallel bubble sort algorithm does not do better than the sequential algorithm. In fact, it is actually slower in most cases. This is because the bubble sort algorithm is a very simple algorithm that does not lend itself well to parallelism. The overhead of synchronizing the threads and dividing the array into smaller parts outweighs the benefits of parallel execution.
    */
    return 0;
}