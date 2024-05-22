#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

void matrixMultiplication(int **mat1, int **mat2, int **result, int row1, int col1, int row2, int col2) {
    #pragma omp parallel num_threads(100)
    {
    #pragma omp for
    for (int i = 0; i < row1; ++i) {
        for (int j = 0; j < col2; ++j) {
            result[i][j] = 0;
            for (int k = 0; k < col1; ++k) {
                result[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }}
}

void displayMatrix(int **matrix, int rows, int cols) {
    #pragma omp parallel num_threads(100)
    {
      #pragma omp for
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }}
}

int main() {
    int row1, col1, row2, col2;

    printf("Enter the number of rows and columns for the first matrix: ");
    scanf("%d %d", &row1, &col1);

    printf("Enter the number of rows and columns for the second matrix: ");
    scanf("%d %d", &row2, &col2);

    if (col1 != row2) {
        printf("Matrix multiplication is not possible.\n");
        return 0;
    }

    int **mat1 = (int **)malloc(row1 * sizeof(int *));
    int **mat2 = (int **)malloc(row2 * sizeof(int *));
    int **result = (int **)malloc(row1 * sizeof(int *));
    int i, j;

    for (i = 0; i < row1; ++i) {
        mat1[i] = (int *)malloc(col1 * sizeof(int));
        result[i] = (int *)malloc(col2 * sizeof(int));
        for (j = 0; j < col1; ++j) {
            mat1[i][j] = rand() % 10; 
        }
    }

    for (i = 0; i < row2; ++i) {
        mat2[i] = (int *)malloc(col2 * sizeof(int));
        for (j = 0; j < col2; ++j) {
            mat2[i][j] = rand() % 10; 
        }
    }


    clock_t start_time = clock();

    matrixMultiplication(mat1, mat2, result, row1, col1, row2, col2);

    clock_t end_time = clock();
    double execution_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("Resultant Matrix:\n");
    displayMatrix(result, row1, col2);
    printf("Execution Time: %.6f seconds\n", execution_time);

  
    for (i = 0; i < row1; ++i) {
        free(mat1[i]);
        free(result[i]);
    }
    for (i = 0; i < row2; ++i) {
        free(mat2[i]);
    }
    free(mat1);
    free(mat2);
    free(result);

    return 0;
}

