#include "CUSolver_helper.h"

void csr2denseMatrixFileInput(char const* FileName, int *lda, int *rowsA, int *colsA, double** h_A, FILE *log) {
    FILE* inA;
    int nnzA = 0, baseA;
    int* h_RowPtrA, * h_ColIndA;
    double start, stop;
    double* h_ValA;
    start = second();
    inA = fopen(FileName, "r");
    if (inA == NULL) {
        printf("Can't read matrix");
        exit(EXIT_FAILURE);
    }
    fscanf(inA, "%d", rowsA);
    
    fscanf(inA, "%d", &nnzA);
    
    h_RowPtrA = (int*)malloc(sizeof(int) * (*rowsA + 1));
    h_ColIndA = (int*)malloc(sizeof(int) * (nnzA));
    h_ValA = (double*)malloc(sizeof(double) * (nnzA));
    
    for (int i = 0; i < *rowsA + 1; ++i) {
        fscanf(inA, "%d", &h_RowPtrA[i]);
    }
    for (int i = 0; i < nnzA; ++i) {
        fscanf(inA, "%d", &h_ColIndA[i]);
    }
    for (int i = 0; i < nnzA; ++i) {
        fscanf(inA, "%lf", &h_ValA[i]);
    }
    
    fclose(inA);
    inA = NULL;

    for (int i = 0; i < nnzA; ++i) {
        if (h_ColIndA[i] + 1 > *colsA) {
            *colsA = h_ColIndA[i] + 1;
        }
    }
    
    stop = second();
    fprintf(log, "A: input                 --- %10.6f sec\n", stop - start);

    start = second();

    *lda = *rowsA;
    *h_A = (double*)malloc(sizeof(double) * (*lda) * (*colsA));
    memset(*h_A, 0, sizeof(double) * (*lda) * (*colsA));

    stop = second();
    fprintf(log, "A: malloc                --- %10.6f sec\n", stop - start);

    baseA = h_RowPtrA[0];  // baseA = {0,1}

    // Convert CSR matrix to dense
    start = second();
    for (int row = 0; row < *rowsA; row++) {
        const int start = h_RowPtrA[row] - baseA;
        const int end = h_RowPtrA[row + 1] - baseA;
        for (int colidx = start; colidx < end; colidx++) {
            const int col = h_ColIndA[colidx] - baseA;
            double Areg = h_ValA[colidx];
            *(*h_A+row + col * (*lda)) = Areg;
        }
    }
    
    stop = second();
    fprintf(log, "A: convert               --- %10.6f sec\n", stop - start);

    free(h_RowPtrA);
    free(h_ColIndA);
    free(h_ValA);

    h_RowPtrA = NULL;
    h_ColIndA = NULL;
    h_ValA = NULL;
}

void denseVectorFileInput(char const* FileName, int* rowsA, double** h_ValA) {
    FILE* inA;
    inA = fopen(FileName, "r");
    if (inA == NULL) {
        printf("Can't read vector");
        exit(EXIT_FAILURE);
    }
    fscanf(inA, "%d", rowsA);
    *h_ValA = (double*)malloc(sizeof(double) * (*rowsA));
    for (int row = 0; row < *rowsA; row++) {
        fscanf(inA, "%lf", *h_ValA + row);
    }
    fclose(inA);
    inA = NULL;
}

void denseVectorFileOutput(char const* FileName, int rowsA, const double* h_ValA) {
    FILE* inA;
    inA = fopen(FileName, "w");
    fprintf(inA, "%d\n", rowsA);
    for (int row = 0; row < rowsA; row++) {
        fprintf(inA, "%e\n", h_ValA[row]);
    }
    fclose(inA);
    inA = NULL;
}