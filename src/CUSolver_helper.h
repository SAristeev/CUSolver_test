#pragma once

#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

void csr2denseMatrixFileInput(char const* FileName, int* lda, int* rowsA, int* colsA, double** h_A, FILE* log);
void denseVectorFileInput(char const* FileName, int* rowsA, double** h_ValA);
void denseVectorFileOutput(char const* FileName, int rowsA, const double* h_ValA);

int linearSolverCHOL(cusolverDnHandle_t handle, int n, const double* Acopy, int lda, const double* b, double* x, FILE* log);
int linearSolverLU  (cusolverDnHandle_t handle, int n, const double* Acopy, int lda, const double* b, double* x, FILE* log);
int linearSolverQR  (cusolverDnHandle_t handle, int n, const double* Acopy, int lda, const double* b, double* x, FILE* log);

double second(void);

double vec_norminf(int n, const double* x);
double mat_norminf(int m, int n, const double* A, int lda);
double csr_mat_norminf(int m, int n, int nnzA, const double* csrValA, const int* csrRowPtrA, const int* csrColIndA);

void testFidesys(const int colsA, const double* h_x, const double* fid_x, FILE * log);
void check(cusolverStatus_t result, char const* const func, const char* const file, int const line);
void check(cublasStatus_t result, char const* const func, const char* const file, int const line);
void check(cudaError result, char const* const func, const char* const file, int const line);