#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

// A highly unrolled 6x8 micro-kernel. 
// A is Column-Major. B and C are Row-Major.
void kernel
(
  int              m,
  int              n,
  int              k,
  const double* restrict a,
  const double* restrict b,
  double* restrict c
){
    // Allocate 12 accumulators mapped to YMM registers
    __m256d c0_0 = _mm256_setzero_pd(); __m256d c0_1 = _mm256_setzero_pd(); 
    __m256d c1_0 = _mm256_setzero_pd(); __m256d c1_1 = _mm256_setzero_pd(); 
    __m256d c2_0 = _mm256_setzero_pd(); __m256d c2_1 = _mm256_setzero_pd(); 
    __m256d c3_0 = _mm256_setzero_pd(); __m256d c3_1 = _mm256_setzero_pd(); 
    __m256d c4_0 = _mm256_setzero_pd(); __m256d c4_1 = _mm256_setzero_pd(); 
    __m256d c5_0 = _mm256_setzero_pd(); __m256d c5_1 = _mm256_setzero_pd(); 

    // Main accumulation loop using outer products
    for (int p = 0; p < k; ++p) {
        
        // Load 1 row of B (8 elements = 2 SIMD vectors)
        __m256d b0 = _mm256_loadu_pd(&b[p * n + 0]);
        __m256d b1 = _mm256_loadu_pd(&b[p * n + 4]);

        // Broadcast scalar A elements (Column-Major: p * m + row) 
        // and multiply-add against B
        __m256d a0 = _mm256_set1_pd(a[p * m + 0]);
        c0_0 = _mm256_fmadd_pd(a0, b0, c0_0);
        c0_1 = _mm256_fmadd_pd(a0, b1, c0_1);

        __m256d a1 = _mm256_set1_pd(a[p * m + 1]);
        c1_0 = _mm256_fmadd_pd(a1, b0, c1_0);
        c1_1 = _mm256_fmadd_pd(a1, b1, c1_1);

        __m256d a2 = _mm256_set1_pd(a[p * m + 2]);
        c2_0 = _mm256_fmadd_pd(a2, b0, c2_0);
        c2_1 = _mm256_fmadd_pd(a2, b1, c2_1);

        __m256d a3 = _mm256_set1_pd(a[p * m + 3]);
        c3_0 = _mm256_fmadd_pd(a3, b0, c3_0);
        c3_1 = _mm256_fmadd_pd(a3, b1, c3_1);

        __m256d a4 = _mm256_set1_pd(a[p * m + 4]);
        c4_0 = _mm256_fmadd_pd(a4, b0, c4_0);
        c4_1 = _mm256_fmadd_pd(a4, b1, c4_1);

        __m256d a5 = _mm256_set1_pd(a[p * m + 5]);
        c5_0 = _mm256_fmadd_pd(a5, b0, c5_0);
        c5_1 = _mm256_fmadd_pd(a5, b1, c5_1);
    }

    // Accumulate results back into the C matrix (Row-Major: row * n + col)
    _mm256_storeu_pd(&c[0 * n + 0], _mm256_add_pd(_mm256_loadu_pd(&c[0 * n + 0]), c0_0));
    _mm256_storeu_pd(&c[0 * n + 4], _mm256_add_pd(_mm256_loadu_pd(&c[0 * n + 4]), c0_1));
    
    _mm256_storeu_pd(&c[1 * n + 0], _mm256_add_pd(_mm256_loadu_pd(&c[1 * n + 0]), c1_0));
    _mm256_storeu_pd(&c[1 * n + 4], _mm256_add_pd(_mm256_loadu_pd(&c[1 * n + 4]), c1_1));
    
    _mm256_storeu_pd(&c[2 * n + 0], _mm256_add_pd(_mm256_loadu_pd(&c[2 * n + 0]), c2_0));
    _mm256_storeu_pd(&c[2 * n + 4], _mm256_add_pd(_mm256_loadu_pd(&c[2 * n + 4]), c2_1));
    
    _mm256_storeu_pd(&c[3 * n + 0], _mm256_add_pd(_mm256_loadu_pd(&c[3 * n + 0]), c3_0));
    _mm256_storeu_pd(&c[3 * n + 4], _mm256_add_pd(_mm256_loadu_pd(&c[3 * n + 4]), c3_1));
    
    _mm256_storeu_pd(&c[4 * n + 0], _mm256_add_pd(_mm256_loadu_pd(&c[4 * n + 0]), c4_0));
    _mm256_storeu_pd(&c[4 * n + 4], _mm256_add_pd(_mm256_loadu_pd(&c[4 * n + 4]), c4_1));

    _mm256_storeu_pd(&c[5 * n + 0], _mm256_add_pd(_mm256_loadu_pd(&c[5 * n + 0]), c5_0));
    _mm256_storeu_pd(&c[5 * n + 4], _mm256_add_pd(_mm256_loadu_pd(&c[5 * n + 4]), c5_1));
}