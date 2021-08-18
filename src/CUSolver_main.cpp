#include "CUSolver_helper.h"

int main(void) {

    double start, stop;
    FILE* log;
    log = fopen("../log/CUSolver.log", "w");

    int lda, rowsA, colsA, rowsB, rowsX;

    double* h_A = NULL;  // dense matrix from CSR(A)
    double* h_x = NULL;  // a copy of d_x
    double* h_b = NULL;  // b = ones(m,1)
    double* h_r = NULL;  // r = b - A*x, a copy of d_r
    double* fid_x = NULL;  // x by Fidesys

    double* d_A = NULL;  // a copy of h_A
    double* d_x = NULL;  // x = A \ b
    double* d_b = NULL;  // a copy of h_b
    double* d_r = NULL;  // r = b - A*x

  // the constants are used in residual evaluation, r = b - A*x
    const double minus_one = -1.0;
    const double one = 1.0;

//==========================================================================
// Input
//==========================================================================

    csr2denseMatrixFileInput("../input/A.txt", &lda, &rowsA, &colsA, &h_A, log);
    denseVectorFileInput("../input/B.vec", &rowsB, &h_b);
    denseVectorFileInput("../input/X.vec", &rowsX, &fid_x);

    fprintf(log, "|A|                      --- %e\n", mat_norminf(rowsA, colsA, h_A, lda));
    fprintf(log, "|Fidesys_X|              --- %e\n\n", vec_norminf(colsA, fid_x));

    h_r = (double*)malloc(sizeof(double) * colsA);
    h_x = (double*)malloc(sizeof(double) * colsA);

//==========================================================================
// CUSolver API Start
//==========================================================================

    cusolverDnHandle_t handle = NULL;
    cublasHandle_t cublasHandle = NULL;  // used in residual evaluation
    cudaStream_t stream = NULL;

    checkCudaErrors(cusolverDnCreate(&handle));
    checkCudaErrors(cublasCreate(&cublasHandle));
    checkCudaErrors(cudaStreamCreate(&stream));

    checkCudaErrors(cusolverDnSetStream(handle, stream));
    checkCudaErrors(cublasSetStream(cublasHandle, stream));

    start = second();

    checkCudaErrors(cudaMalloc((void**)&d_A, sizeof(double) * lda * colsA));
    checkCudaErrors(cudaMalloc((void**)&d_x, sizeof(double) * colsA));
    checkCudaErrors(cudaMalloc((void**)&d_b, sizeof(double) * colsA));
    checkCudaErrors(cudaMalloc((void**)&d_r, sizeof(double) * colsA));

    checkCudaErrors(cudaMemcpy(d_A, h_A, sizeof(double) * lda * colsA, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, h_b, sizeof(double) * colsA, cudaMemcpyHostToDevice));

    stop = second();
    fprintf(log, "CUDA Malloc + Memcpy     --- %10.6f sec\n\n", stop - start);

//==========================================================================
// First Solver
//==========================================================================

    linearSolverCHOL(handle, rowsA, d_A, lda, d_b, d_x, log);

    checkCudaErrors(cudaMemcpy(d_r, d_b, sizeof(double) * rowsA, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cublasDgemm_v2(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, rowsA,
        1, colsA, &minus_one, d_A, lda, d_x, rowsA,
        &one, d_r, rowsA));

    checkCudaErrors(cudaMemcpy(h_x, d_x, sizeof(double) * colsA, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_r, d_r, sizeof(double) * rowsA, cudaMemcpyDeviceToHost));
    
    testFidesys(colsA, h_x, fid_x, log);
    fprintf(log, "|CHOL(X)|                --- %e\n", vec_norminf(colsA, h_x));
    fprintf(log, "|b - A*x|                --- %e\n\n", vec_norminf(colsA, h_r));
        
    denseVectorFileOutput("../output/X_CHOL.vec", colsA, h_x);
    
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_r));

//==========================================================================
// Second Solver
//==========================================================================

    checkCudaErrors(cudaMalloc((void**)&d_x, sizeof(double) * colsA));
    checkCudaErrors(cudaMalloc((void**)&d_r, sizeof(double) * colsA));

    linearSolverLU(handle, rowsA, d_A, lda, d_b, d_x, log);

    checkCudaErrors(cudaMemcpy(d_r, d_b, sizeof(double) * rowsA, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cublasDgemm_v2(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, rowsA,
        1, colsA, &minus_one, d_A, lda, d_x, rowsA,
        &one, d_r, rowsA));

    checkCudaErrors(cudaMemcpy(h_x, d_x, sizeof(double) * colsA, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_r, d_r, sizeof(double) * rowsA, cudaMemcpyDeviceToHost));

    testFidesys(colsA, h_x, fid_x, log);
    fprintf(log, "|LU(X)|                  --- %e\n", vec_norminf(colsA, h_x));
    fprintf(log, "|b - A*x|                --- %e\n\n", vec_norminf(colsA, h_r));
    denseVectorFileOutput("../output/X_LU.vec", colsA, h_x);

    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_r));

//==========================================================================
// Third Solver
//==========================================================================

    checkCudaErrors(cudaMalloc((void**)&d_x, sizeof(double)* colsA));
    checkCudaErrors(cudaMalloc((void**)&d_r, sizeof(double)* colsA));

    linearSolverQR(handle, rowsA, d_A, lda, d_b, d_x, log);

    checkCudaErrors(cudaMemcpy(d_r, d_b, sizeof(double)* rowsA, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cublasDgemm_v2(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, rowsA,
        1, colsA, &minus_one, d_A, lda, d_x, rowsA,
        &one, d_r, rowsA));

    checkCudaErrors(cudaMemcpy(h_x, d_x, sizeof(double)* colsA, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_r, d_r, sizeof(double)* rowsA, cudaMemcpyDeviceToHost));

    testFidesys(colsA, h_x, fid_x, log);
    fprintf(log, "|QR(X)|                  --- %e\n", vec_norminf(colsA, h_x));
    fprintf(log, "|b - A*x|                --- %e\n\n", vec_norminf(colsA, h_r));
    denseVectorFileOutput("../output/X_QR.vec", colsA, h_x);


    free(h_x);
    free(h_r);

    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_r));

    checkCudaErrors(cusolverDnDestroy(handle));
    checkCudaErrors(cublasDestroy(cublasHandle));
    
    checkCudaErrors(cudaStreamDestroy(stream));

    free(h_A);
    free(h_b);
    
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_b));
   

    fclose(log);
    log = NULL;

    printf("Done\nSee info on 'log/CUSolver.log'\n");

    return 0;
}