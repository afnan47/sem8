%%cu
#include <iostream>
#include <math.h>
#include <functional>
#include <stdlib.h>    
#include <time.h>       

#define ROW_TILE_WIDTH 32
#define COL_TILE_WIDTH 32

template<typename T>
__global__
void naive_matrix_multiply(T *A, T *B, T* C, int width, int C_rows, int C_cols)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;   
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  // check boundry conditions
  if( row < C_rows && col < C_cols ){
    // do the multiplication for one row and col
    T value = 0;
    for(int k = 0; k < width; k++){
      value += A[row * width + k] * B[k * C_cols + col];
    }
    // store result
    C[row * C_cols + col] = value;
  }
  

}

template<typename T>
void initialize_matrix(T* M, int rows, int cols, std::function<float()> F) {
  for(int i = 0; i < rows; i++){
    for(int j = 0; j < cols; j++){
      M[i * cols + j] = F();
    }
  }
}

template<typename T>
void naive_matrix_multiply_cpu(T *A, T *B, T* C, int width, int C_rows, int C_cols){
  for(int i = 0; i < C_rows; i++)
    for(int j = 0; j < C_cols; j++){
      T value = 0.0f;
      for(int k = 0; k < width; k++){
        value += A[i * width + k] * B[k * C_cols + j];
      }
      C[i * C_cols + j] = value;
    }
}

template<typename T>
bool check_equal(T* A1, T* A2, int rows, int cols){
  for(int i = 0; i < rows; i++)
    for(int j = 0; j < cols; j++){
      if(abs(A1[i * cols + j] - A2[i * cols + j]) > 0.00001){
          return false;
      }
    }
  
  return true;
}


int main(void)
{
  int A_rows = 1 << 8;
  int A_cols = 1 << 10;
  int B_rows = A_cols;
  int B_cols = 1 << 12;
  int C_rows = A_rows;
  int C_cols = B_cols;
  int A_size = A_rows * A_cols;
  int B_size = B_rows * B_cols;
  int C_size = C_rows * C_cols;
  float *A, *B, *C, *C_cpu;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&A, A_size*sizeof(float));
  cudaMallocManaged(&B, B_size*sizeof(float));
  cudaMallocManaged(&C, C_size*sizeof(float));
  cudaMallocManaged(&C_cpu, C_size*sizeof(float));

  // initialize A and B matrices
  auto all_ones = []() -> float {
    return 1.0f;
  };

  srand (time(NULL));
  auto rand_numbers = []() -> float {
    auto f = static_cast<float>(rand())/(static_cast<float>(RAND_MAX/1000));
    int n = static_cast<int>(f);
    return static_cast<float>(n);
  };

  initialize_matrix<float>(A, A_rows, A_cols, rand_numbers);
  initialize_matrix<float>(B, B_rows, B_cols, rand_numbers);

  dim3 dim_grid(C_cols/COL_TILE_WIDTH, C_rows/ROW_TILE_WIDTH, 1);
  dim3 dim_block(COL_TILE_WIDTH, ROW_TILE_WIDTH, 1);

  naive_matrix_multiply<float><<<dim_grid, dim_block>>>(A, B, C, A_cols, C_rows, C_cols);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  
  // check results
  naive_matrix_multiply_cpu<float>(A, B, C_cpu, A_cols, C_rows, C_cols);
  
  
  if(check_equal<float>(C, C_cpu, C_rows, C_cols))
    std::cout << "PASS" << std::endl;
  else
    std::cout << "FAIL" << std::endl;

  // Free memory
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  
  return 0; 
}
/* 
Set up CUDA on collab to run this code.
First Change runtime to GPU and run these codes:
!pip install git+https://github.com/andreinechaev/nvcc4jupyter.git
%load_ext nvcc_plugin
*/