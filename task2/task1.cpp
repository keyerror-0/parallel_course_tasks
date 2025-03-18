#include <iostream>
#include <vector>
#include <omp.h>
int main(){
    size_t n = 5;
    std::vector<std::vector<double>> matrix(n, std::vector<double>(n));
    std::vector<double> vector(n);
    std::vector<double> result(n);

    for(int i = 0; i < n; i++){
        result[i] = 0;
        vector[i] = i;
        for(int j = 0; j < n; j++){
            matrix[i][j] = i*n +j;
        }
    }



    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            result[i] += matrix[i][j] * vector[j];
        }

        std::cout << result[i] << std::endl;
    }

    #ifdef _OPENMP
#pragma omp parallel num_threads(6)
    {
        printf("Hello, multithreaded world: thread %d of %d\n",
               omp_get_thread_num(), omp_get_num_threads());
    }
    printf("OpenMP version %d\n", _OPENMP);
    if (_OPENMP >= 201107)
        printf(" OpenMP 3.1 is supported\n");
#endif
    return 0;


    return 0;
}

