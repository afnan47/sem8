#include <iostream>
#include <cmath>
#include <chrono>
#include <omp.h>
using namespace std;

const int ARRAY_SIZE = 100000000;

int main() {
    double* array = new double[ARRAY_SIZE];
    double min = numeric_limits<double>::max();
    double max = numeric_limits<double>::min();
    double sum = 0.0;
    double avg = 0.0;

    // Initialize array with random values
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        array[i] = rand();
    }

    // Sequential version
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        if (array[i] < min) {
            min = array[i];
        }
        if (array[i] > max) {
            max = array[i];
        }
        sum += array[i];
    }
    avg = sum / ARRAY_SIZE;
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed_seconds = end - start;

    cout << "Sequential version:" << endl;
    cout << "Min: " << min << endl;
    cout << "Max: " << max << endl;
    cout << "Sum: " << sum << endl;
    cout << "Average: " << avg << endl;
    cout << "Elapsed time: " << elapsed_seconds.count() << "s" << endl;

    // Parallel version using OpenMP
    min = numeric_limits<double>::max();
    max = numeric_limits<double>::min();
    sum = 0.0;
    avg = 0.0;

    start = chrono::high_resolution_clock::now();
#pragma omp parallel for reduction(min:min) reduction(max:max) reduction(+:sum)
{
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        if (array[i] < min) {
            min = array[i];
        }
        if (array[i] > max) {
            max = array[i];
        }
        sum += array[i];
    }
}
    avg = sum / ARRAY_SIZE;
    end = chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;

    cout << "Parallel version:" << endl;
    cout << "Min: " << min << endl;
    cout << "Max: " << max << endl;
    cout << "Sum: " << sum << endl;
    cout << "Average: " << avg << endl;
    cout << "Elapsed time: " << elapsed_seconds.count() << "s" << endl;

    delete[] array;
    return 0;
}
