/*
 *  Test three linear solvers, including Cholesky, LU and QR.
 *  The program solves
 *          A*x = b  where b = ones(m,1)
 *  and reports relative error
 *          |b-A*x|/(|A|*|x|)
 *
 *  The elapsed time is also reported so the user can compare efficiency of
 * different solvers.
 *
 *  Remark: the absolute error on solution x is meaningless without knowing
 * condition number of A. The relative error on residual should be close to
 * machine zero, i.e. 1.e-15.
 */

#pragma warning( push )
#pragma warning( disable : 26812 )
#include <assert.h>
 #include <ctype.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>

#include <cuda_runtime.h>

#include "cublas_v2.h"
#include "cusolverDn.h"
#include "helper_cuda.h"

#include "helper_cusolver.h"

/*
 *  solve A*x = b by Cholesky factorization
 *
 */
int linearSolverCHOL(cusolverDnHandle_t handle, int n, const double* Acopy,
    int lda, const double* b, double* x) {
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
    fprintf(stdout, "timing: cholesky = %10.6f sec\n", time_solve);

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

/*
 *  solve A*x = b by LU with partial pivoting
 *
 */
int linearSolverLU(cusolverDnHandle_t handle, int n, const double* Acopy,
    int lda, const double* b, double* x) {
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
    fprintf(stdout, "timing: LU = %10.6f sec\n", time_solve);

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

/*
 *  solve A*x = b by QR
 *
 */
int linearSolverQR(cusolverDnHandle_t handle, int n, const double* Acopy,
    int lda, const double* b, double* x) {
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
    fprintf(stdout, "timing: QR = %10.6f sec\n", time_solve);

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

int main(void) {
    double start, stop, all_time_solve;

    cusolverDnHandle_t handle = NULL;
    cublasHandle_t cublasHandle = NULL;  // used in residual evaluation
    cudaStream_t stream = NULL;
    int rowsA = 0;  // number of rows of A
    int colsA = 0;  // number of columns of A
    int nnzA = 0;   // number of nonzeros of A
    int baseA = 0;  // base index in CSR format
    int lda = 0;    // leading dimension in dense matrix
    int rowsB = 0;
    int rowsX = 0;

    // CSR(A) from I/O
    int* h_csrRowPtrA = NULL;
    int* h_csrColIndA = NULL;
    double* h_csrValA = NULL;

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

    double x_inf = 0.0;
    double r_inf = 0.0;
    double A_inf = 0.0;
    double fid_inf = 0.0;
    int errors = 0;


    errno_t errInputA, errInputB, errInputX;

    //==  Input CSR matrix and vector  ==// 

    FILE* inA, * inB, * inX;
    errInputA = fopen_s(&inA, "../input/A.txt", "r");
    if (errInputA) {
        printf("Can't read matrix");
        return -1;
    }
    errInputB = fopen_s(&inB, "../input/B.vec", "r");
    if (errInputB) {
        printf("Can't read vector");
        return -1;
    }
    errInputX = fopen_s(&inX, "../input/X.vec", "r");
    if (errInputX) {
        printf("Can't read vector");
        return -1;
    }
    fscanf_s(inA, "%d", &rowsA);
    fscanf_s(inA, "%d", &nnzA);

    h_csrRowPtrA = (int*)malloc(sizeof(int) * (rowsA + 1));
    h_csrColIndA = (int*)malloc(sizeof(int) * nnzA);
    h_csrValA = (double*)malloc(sizeof(double) * nnzA);

    for (int i = 0; i < rowsA + 1; ++i) {
        fscanf_s(inA, "%d", &h_csrRowPtrA[i]);
    }
    for (int i = 0; i < nnzA; ++i) {
        fscanf_s(inA, "%d", &h_csrColIndA[i]);
    }
    for (int i = 0; i < nnzA; ++i) {
        fscanf_s(inA, "%lf", &h_csrValA[i]);
    }


    for (int i = 0; i < nnzA; ++i) {
        if (h_csrColIndA[i] + 1 > colsA) {
            colsA = h_csrColIndA[i] + 1;
        }
    }

    lda = rowsA > colsA ? rowsA : colsA;

    h_A = (double*)malloc(sizeof(double) * lda * colsA);
    h_x = (double*)malloc(sizeof(double) * colsA);
    h_b = (double*)malloc(sizeof(double) * rowsA);
    h_r = (double*)malloc(sizeof(double) * rowsA);
    fid_x = (double*)malloc(sizeof(double) * colsA);

    assert(NULL != h_A);
    assert(NULL != h_x);
    assert(NULL != h_b);
    assert(NULL != h_r);
    assert(NULL != fid_x);

    baseA = h_csrRowPtrA[0];  // baseA = {0,1}

    memset(h_A, 0, sizeof(double) * colsA * lda);

    // Convert CSR matrix to dense

    for (int row = 0; row < rowsA; row++) {
        const int start = h_csrRowPtrA[row] - baseA;
        const int end = h_csrRowPtrA[row + 1] - baseA;
        for (int colidx = start; colidx < end; colidx++) {
            const int col = h_csrColIndA[colidx] - baseA;
            const double Areg = h_csrValA[colidx];
            h_A[row + col * lda] = Areg;
        }
    }

    fscanf_s(inB, "%d", &rowsB);
    for (int row = 0; row < rowsB; row++) {
        fscanf_s(inB, "%lf", &h_b[row]);
    }
    fscanf_s(inX, "%d", &rowsX);
    for (int row = 0; row < rowsX; row++) {
        fscanf_s(inX, "%lf", &fid_x[row]);
    }
    /*for (int row = 0; row < rowsB; row++) {
        printf("%E\n", h_b[row]);
    }*/
    //printf("%d %d", rowsA, colsA);
    //printMatrix(rowsA, colsA, h_A, lda);
    start = second();
    checkCudaErrors(cusolverDnCreate(&handle));
    checkCudaErrors(cublasCreate(&cublasHandle));
    checkCudaErrors(cudaStreamCreate(&stream));

    checkCudaErrors(cusolverDnSetStream(handle, stream));
    checkCudaErrors(cublasSetStream(cublasHandle, stream));

    checkCudaErrors(cudaMalloc((void**)&d_A, sizeof(double) * lda * colsA));
    checkCudaErrors(cudaMalloc((void**)&d_x, sizeof(double) * colsA));
    checkCudaErrors(cudaMalloc((void**)&d_b, sizeof(double) * rowsA));
    checkCudaErrors(cudaMalloc((void**)&d_r, sizeof(double) * rowsA));
    checkCudaErrors(cudaMemcpy(d_A, h_A, sizeof(double) * lda * colsA,
        cudaMemcpyHostToDevice));
    checkCudaErrors(
        cudaMemcpy(d_b, h_b, sizeof(double) * rowsA, cudaMemcpyHostToDevice));
    linearSolverCHOL(handle, rowsA, d_A, lda, d_b, d_x);
    checkCudaErrors(
        cudaMemcpy(d_r, d_b, sizeof(double) * rowsA, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cublasDgemm_v2(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, rowsA,
        1, colsA, &minus_one, d_A, lda, d_x, rowsA,
        &one, d_r, rowsA));

    checkCudaErrors(
        cudaMemcpy(h_x, d_x, sizeof(double) * colsA, cudaMemcpyDeviceToHost));
    checkCudaErrors(
        cudaMemcpy(h_r, d_r, sizeof(double) * rowsA, cudaMemcpyDeviceToHost));
    stop = second();
    all_time_solve = stop - start;
    printf("all CUDA timing: = % 10.6f sec\n", all_time_solve);
    /*for (int col = 0; col < colsA; col++) {
        printf("%E\n",h_x[col]);
    }*/

    for (int row = 0; row < rowsX; row++) {
        fid_x[row] -= h_x[row];
    }

    x_inf = vec_norminf(colsA, h_x);
    r_inf = vec_norminf(rowsA, h_r);
    A_inf = mat_norminf(rowsA, colsA, h_A, lda);
    fid_inf = vec_norminf(rowsA, fid_x);

    printf("|b - A*x| = %E \n", r_inf);
    printf("|A| = %E \n", A_inf);
    printf("|x| = %E \n", x_inf);
    printf("|Fid_x-x| = %E \n", fid_inf);
    printf("|b - A*x|/(|A|*|x|) = %E \n", r_inf / (A_inf * x_inf));

    if (handle) {
        checkCudaErrors(cusolverDnDestroy(handle));
    }
    if (cublasHandle) {
        checkCudaErrors(cublasDestroy(cublasHandle));
    }
    if (stream) {
        checkCudaErrors(cudaStreamDestroy(stream));
    }

    if (h_csrValA) {
        free(h_csrValA);
    }
    if (h_csrRowPtrA) {
        free(h_csrRowPtrA);
    }
    if (h_csrColIndA) {
        free(h_csrColIndA);
    }

    if (h_A) {
        free(h_A);
    }
    if (h_x) {
        free(h_x);
    }
    if (h_b) {
        free(h_b);
    }
    if (h_r) {
        free(h_r);
    }

    if (d_A) {
        checkCudaErrors(cudaFree(d_A));
    }
    if (d_x) {
        checkCudaErrors(cudaFree(d_x));
    }
    if (d_b) {
        checkCudaErrors(cudaFree(d_b));
    }
    if (d_r) {
        checkCudaErrors(cudaFree(d_r));
    }

    return 0;
}
#pragma warning( pop )