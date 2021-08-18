#include "CUSolver_helper.h"

static const char* _cudaGetErrorEnum(cudaError_t error) {
    return cudaGetErrorName(error);
}

static const char* _cudaGetErrorEnum(cublasStatus_t error) {
    switch (error) {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
    }

    return "<unknown>";
}


static const char* _cudaGetErrorEnum(cusolverStatus_t error) {
    switch (error) {
    case CUSOLVER_STATUS_SUCCESS:
        return "CUSOLVER_STATUS_SUCCESS";
    case CUSOLVER_STATUS_NOT_INITIALIZED:
        return "CUSOLVER_STATUS_NOT_INITIALIZED";
    case CUSOLVER_STATUS_ALLOC_FAILED:
        return "CUSOLVER_STATUS_ALLOC_FAILED";
    case CUSOLVER_STATUS_INVALID_VALUE:
        return "CUSOLVER_STATUS_INVALID_VALUE";
    case CUSOLVER_STATUS_ARCH_MISMATCH:
        return "CUSOLVER_STATUS_ARCH_MISMATCH";
    case CUSOLVER_STATUS_MAPPING_ERROR:
        return "CUSOLVER_STATUS_MAPPING_ERROR";
    case CUSOLVER_STATUS_EXECUTION_FAILED:
        return "CUSOLVER_STATUS_EXECUTION_FAILED";
    case CUSOLVER_STATUS_INTERNAL_ERROR:
        return "CUSOLVER_STATUS_INTERNAL_ERROR";
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    case CUSOLVER_STATUS_NOT_SUPPORTED:
        return "CUSOLVER_STATUS_NOT_SUPPORTED ";
    case CUSOLVER_STATUS_ZERO_PIVOT:
        return "CUSOLVER_STATUS_ZERO_PIVOT";
    case CUSOLVER_STATUS_INVALID_LICENSE:
        return "CUSOLVER_STATUS_INVALID_LICENSE";
    }

    return "<unknown>";
}







void check(cusolverStatus_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
        exit(EXIT_FAILURE);
    }
}
void check(cublasStatus_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
        exit(EXIT_FAILURE);
    }
}
void check(cudaError result, char const* const func, const char* const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
        exit(EXIT_FAILURE);
    }
}

double vec_norminf(int n, const double* x)
{
    double norminf = 0;
    for (int j = 0; j < n; j++) {
        double x_abs = fabs(x[j]);
        norminf = (norminf > x_abs) ? norminf : x_abs;
    }
    return norminf;
}

double mat_norminf(
    int m,
    int n,
    const double* A,
    int lda)
{
    double norminf = 0;
    for (int i = 0; i < m; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            double A_abs = fabs(A[i + j * lda]);
            sum += A_abs;
        }
        norminf = (norminf > sum) ? norminf : sum;
    }
    return norminf;
}

/*
 * |A| = max { |A|*ones(m,1) }
 */
double csr_mat_norminf(
    int m,
    int n,
    int nnzA,
    const double* csrValA,
    const int* csrRowPtrA,
    const int* csrColIndA)
{
    const int baseA = 0; // or 1

    double norminf = 0;
    for (int i = 0; i < m; i++) {
        double sum = 0.0;
        const int start = csrRowPtrA[i] - baseA;
        const int end = csrRowPtrA[i + 1] - baseA;
        for (int colidx = start; colidx < end; colidx++) {
            double A_abs = fabs(csrValA[colidx]);
            sum += A_abs;
        }
        norminf = (norminf > sum) ? norminf : sum;
    }
    return norminf;
}

void testFidesys(const int colsA, const double* h_x, const double* fid_x, FILE * log)
{
    FILE* DIFF;
    DIFF = fopen("../output/diff_CHOL.txt", "w");
    double* h_difference;
    h_difference = (double*)malloc(sizeof(double) * colsA);
    for (int i = 0; i < colsA; ++i) {
        h_difference[i] = h_x[i] - fid_x[i];
        fprintf(DIFF, "%e\n", h_difference[i]);
    }
    fclose(DIFF);
    DIFF = NULL;
    fprintf(log, "inf-norm of |difference| --- %e\n", vec_norminf(colsA, h_difference));
}

#if defined(_WIN32)
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
double second(void)
{
    LARGE_INTEGER t;
    static double oofreq;
    static int checkedForHighResTimer;
    static BOOL hasHighResTimer;

    if (!checkedForHighResTimer) {
        hasHighResTimer = QueryPerformanceFrequency(&t);
        oofreq = 1.0 / (double)t.QuadPart;
        checkedForHighResTimer = 1;
    }
    if (hasHighResTimer) {
        QueryPerformanceCounter(&t);
        return (double)t.QuadPart * oofreq;
    }
    else {
        return (double)GetTickCount() / 1000.0;
    }
}

#elif defined(__linux__) || defined(__QNX__)
#include <stddef.h>
#include <sys/time.h>
#include <sys/resource.h>
double second(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}
#elif defined(__APPLE__)
#include <stddef.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/sysctl.h>
double second(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}
#else
#error unsupported platform
#endif