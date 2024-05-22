#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h> 

void generate_random_vector(int vector[], int size) {
    #pragma omp parallel num_threads(5)
    {
        #pragma omp for
        for (int i = 0; i < size; i++) {
            vector[i] = rand() % 100; 
        }
    }
}


void add_vectors(int v1[], int v2[], int result[], int size) {
    #pragma omp parallel num_threads(5)
    {
        #pragma omp for
        for (int i = 0; i < size; i++) {
            result[i] = v1[i] + v2[i];
        }
    }
}

void print_vector(int vector[], int size) {
    printf("[");
    for (int i = 0; i < size; i++) {
        printf("%d", vector[i]);
        if (i < size - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}

int main() {
   
    srand(time(NULL));

   
    int size;
    printf("Enter the size of the vectors: ");
    scanf("%d", &size);

    
    int *v1 = (int *)malloc(size * sizeof(int));
    int *v2 = (int *)malloc(size * sizeof(int));
    int *result = (int *)malloc(size * sizeof(int));

   
    generate_random_vector(v1, size);
    generate_random_vector(v2, size);

   
    double start = omp_get_wtime();

   
    add_vectors(v1, v2, result, size);

    
    double end = omp_get_wtime();

  
  
    printf("Result (Vector 1 + Vector 2): ");
    print_vector(result, size);

   
    printf("Execution Time: %.6f seconds\n", end - start);

   
   
    free(v1);
    free(v2);
    free(result);

    return 0;
}

