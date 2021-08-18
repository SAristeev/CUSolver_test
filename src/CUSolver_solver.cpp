#include "CUSolver_helper.h"

int linearSolverCHOL(cusolverDnHandle_t handle, int n, const double* Acopy,
    int lda, const double* b, double* x, FILE* log) {
    int bufferSize = 0;
    int* info = NULL;
    double* buffer = NULL;
    double* A = NULL;
    int h_info = 0;
    double start, stop;
    double time_solve;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    checkCudaErrors(cusolverDnDpotrf_bufferSize(handle, uplo, n, (double*)Acopy,
        lda, &bufferSize));

    checkCudaErrors(cudaMalloc(&info, sizeof(int)));
    checkCudaErrors(cudaMalloc(&buffer, sizeof(double) * bufferSize));
    checkCudaErrors(cudaMalloc(&A, sizeof(double) * lda * n));

    // prepare a copy of A because potrf will overwrite A with L
    checkCudaErrors(
        cudaMemcpy(A, Acopy, sizeof(double) * lda * n, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemset(info, 0, sizeof(int)));

    start = second();
    start = second();

    checkCudaErrors(
        cusolverDnDpotrf(handle, uplo, n, A, lda, buffer, bufferSize, info));

    checkCudaErrors(
        cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    if (0 != h_info) {
        fprintf(stderr, "Error: Cholesky factorization failed\n");
    }

    checkCudaErrors(
        cudaMemcpy(x, b, sizeof(double) * n, cudaMemcpyDeviceToDevice));

    checkCudaErrors(cusolverDnDpotrs(handle, uplo, n, 1, A, lda, x, n, info));

    checkCudaErrors(cudaDeviceSynchronize());
    stop = second();

    time_solve = stop - start;

    fprintf(log, "Cholesky Solver\n");
    fprintf(log, "timing: cholesky         --- %10.6f sec\n", time_solve);

    if (info) {
        checkCudaErrors(cudaFree(info));
    }
    if (buffer) {
        checkCudaErrors(cudaFree(buffer));
    }
    if (A) {
        checkCudaErrors(cudaFree(A));
    }

    return 0;
}

int linearSolverLU(cusolverDnHandle_t handle, int n, const double* Acopy,
    int lda, const double* b, double* x, FILE* log) {
    int bufferSize = 0;
    int* info = NULL;
    double* buffer = NULL;
    double* A = NULL;
    int* ipiv = NULL;  // pivoting sequence
    int h_info = 0;
    double start, stop;
    double time_solve;

    checkCudaErrors(cusolverDnDgetrf_bufferSize(handle, n, n, (double*)Acopy,
        lda, &bufferSize));

    checkCudaErrors(cudaMalloc(&info, sizeof(int)));
    checkCudaErrors(cudaMalloc(&buffer, sizeof(double) * bufferSize));
    checkCudaErrors(cudaMalloc(&A, sizeof(double) * lda * n));
    checkCudaErrors(cudaMalloc(&ipiv, sizeof(int) * n));

    // prepare a copy of A because getrf will overwrite A with L
    checkCudaErrors(
        cudaMemcpy(A, Acopy, sizeof(double) * lda * n, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemset(info, 0, sizeof(int)));

    start = second();
    start = second();

    checkCudaErrors(cusolverDnDgetrf(handle, n, n, A, lda, buffer, ipiv, info));
    checkCudaErrors(
        cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    if (0 != h_info) {
        fprintf(stderr, "Error: LU factorization failed\n");
    }

    checkCudaErrors(
        cudaMemcpy(x, b, sizeof(double) * n, cudaMemcpyDeviceToDevice));
    checkCudaErrors(
        cusolverDnDgetrs(handle, CUBLAS_OP_N, n, 1, A, lda, ipiv, x, n, info));
    checkCudaErrors(cudaDeviceSynchronize());
    stop = second();

    time_solve = stop - start;

    fprintf(log, "LU Solver\n");
    fprintf(log, "timing: LU               --- %10.6f sec\n", time_solve);

    if (info) {
        checkCudaErrors(cudaFree(info));
    }
    if (buffer) {
        checkCudaErrors(cudaFree(buffer));
    }
    if (A) {
        checkCudaErrors(cudaFree(A));
    }
    if (ipiv) {
        checkCudaErrors(cudaFree(ipiv));
    }

    return 0;
}

int linearSolverQR(cusolverDnHandle_t handle, int n, const double* Acopy,
    int lda, const double* b, double* x, FILE* log) {
    cublasHandle_t cublasHandle = NULL;  // used in residual evaluation
    int bufferSize = 0;
    int bufferSize_geqrf = 0;
    int bufferSize_ormqr = 0;
    int* info = NULL;
    double* buffer = NULL;
    double* A = NULL;
    double* tau = NULL;
    int h_info = 0;
    double start, stop;
    double time_solve;
    const double one = 1.0;

    checkCudaErrors(cublasCreate(&cublasHandle));

    checkCudaErrors(cusolverDnDgeqrf_bufferSize(handle, n, n, (double*)Acopy,
        lda, &bufferSize_geqrf));
    checkCudaErrors(cusolverDnDormqr_bufferSize(handle, CUBLAS_SIDE_LEFT,
        CUBLAS_OP_T, n, 1, n, A, lda,
        NULL, x, n, &bufferSize_ormqr));

    printf("buffer_geqrf = %d, buffer_ormqr = %d \n", bufferSize_geqrf,
        bufferSize_ormqr);

    bufferSize = (bufferSize_geqrf > bufferSize_ormqr) ? bufferSize_geqrf
        : bufferSize_ormqr;

    checkCudaErrors(cudaMalloc(&info, sizeof(int)));
    checkCudaErrors(cudaMalloc(&buffer, sizeof(double) * bufferSize));
    checkCudaErrors(cudaMalloc(&A, sizeof(double) * lda * n));
    checkCudaErrors(cudaMalloc((void**)&tau, sizeof(double) * n));

    // prepare a copy of A because getrf will overwrite A with L
    checkCudaErrors(
        cudaMemcpy(A, Acopy, sizeof(double) * lda * n, cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaMemset(info, 0, sizeof(int)));

    start = second();
    start = second();

    // compute QR factorization
    checkCudaErrors(
        cusolverDnDgeqrf(handle, n, n, A, lda, tau, buffer, bufferSize, info));

    checkCudaErrors(
        cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    if (0 != h_info) {
        fprintf(stderr, "Error: LU factorization failed\n");
    }

    checkCudaErrors(
        cudaMemcpy(x, b, sizeof(double) * n, cudaMemcpyDeviceToDevice));

    // compute Q^T*b
    checkCudaErrors(cusolverDnDormqr(handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, n, 1,
        n, A, lda, tau, x, n, buffer, bufferSize,
        info));

    // x = R \ Q^T*b
    checkCudaErrors(cublasDtrsm(cublasHandle, CUBLAS_SIDE_LEFT,
        CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
        CUBLAS_DIAG_NON_UNIT, n, 1, &one, A, lda, x, n));
    checkCudaErrors(cudaDeviceSynchronize());
    stop = second();

    time_solve = stop - start;
    fprintf(log, "QR Solver\n");
    fprintf(log, "timing: QR               --- %10.6f sec\n", time_solve);

    if (cublasHandle) {
        checkCudaErrors(cublasDestroy(cublasHandle));
    }
    if (info) {
        checkCudaErrors(cudaFree(info));
    }
    if (buffer) {
        checkCudaErrors(cudaFree(buffer));
    }
    if (A) {
        checkCudaErrors(cudaFree(A));
    }
    if (tau) {
        checkCudaErrors(cudaFree(tau));
    }

    return 0;
}