#include<iostream>
#include<omp.h>
using namespace std;

void merge(int a[], int i1, int j1, int i2, int j2, int n){
    int temp[n];    
    int i, j, k;
    i = i1;    
    j = i2;    
    k = 0;
    while(i <= j1 && j <= j2){
        if(a[i] < a[j]) temp[k++] = a[i++];
        else temp[k++] = a[j++];
	}
    
    while(i <= j1) temp[k++] = a[i++];
    while(j <= j2) temp[k++] = a[j++];
  
    for(i = i1, j = 0; i <= j2; i++, j++) a[i] = temp[j];
}

void mergesort(int a[],int i,int j, int n){
    int mid;
    if(i < j){
        mid = (i + j) / 2;
        mergesort(a, i, mid, n);        
        mergesort(a, mid + 1, j, n);    
        merge(a, i, mid, mid + 1, j, n);    
    }
}

void pMergesort(int a[],int i,int j, int n){
    int mid;
    if(i < j){
        mid = (i + j) / 2;
        #pragma omp task firstprivate(a, i, j, n)
        mergesort(a, i, mid, n);        
        
        #pragma omp task firstprivate(a, i, j, n)
        mergesort(a, mid + 1, j, n);    

        #pragma omp taskwait
        merge(a, i, mid, mid + 1, j, n);    
    }
}

int main()
{
    int n = 10;
    int a[n];
    double start_time, end_time;

    // Create an array of n numbers, with digits from n to 1 in descending order
    for(int i = 0, j = n; i < n; i++, j--) a[i] = j;
    
    // Create a copy 
    int b[n];
    for(int i = 0; i < n; i++) b[i] = a[i];


    // Measure Sequential Time
    start_time = omp_get_wtime(); 
    mergesort(a, 0, n - 1, n);
    end_time = omp_get_wtime(); 
    cout << "Time taken by sequential algorithm: " << end_time - start_time << " seconds\n";

     //Measure Parallel time
    start_time = omp_get_wtime(); 
    #pragma omp parallel
    {
        #pragma omp single
        {
            mergesort(b, 0, n - 1, n);
        }
    }
    end_time = omp_get_wtime(); 
    cout << "\nTime taken by parallel algorithm: " << end_time - start_time << " seconds";

    // Unfortunately parallel algorithms only do well on large scales.
    // In our case sequential may always do better than parallel.
    // This is because parallel algorithms have the overhead of creating threads.
    return 0;
}
