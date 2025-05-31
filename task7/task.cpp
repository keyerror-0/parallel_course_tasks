#include <iostream>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <memory>
#include <fstream>
#include <cublas_v2.h>
#include <nvtx3/nvToolsExt.h>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

// Инициализация сетки и граничных условий
void initialize(double* A, double* Anew, size_t size) {
    memset(A, 0, size * size * sizeof(double));
    memset(Anew, 0, size * size * sizeof(double));
    
    A[0] = 10.0;
    A[size-1] = 20.0;
    A[size*(size-1)] = 30.0;
    A[size*size-1] = 20.0; 
    
    double top_left = A[0];
    double top_right = A[size-1];
    double bottom_left = A[size*(size-1)];
    double bottom_right = A[size*size-1];
    
    // Линейная интерполяция граничных условий
    for (int i = 1; i < size-1; ++i) {
        A[i] = top_left + (top_right - top_left) * i / static_cast<double>(size-1);
        A[size*(size-1) + i] = bottom_left + (bottom_right - bottom_left) * i / static_cast<double>(size-1);
        
        A[size*i] = top_left + (bottom_left - top_left) * i / static_cast<double>(size-1);
        A[size*i + size-1] = top_right + (bottom_right - top_right) * i / static_cast<double>(size-1);
    }
}

// Вычисление новых значений для внутренних точек
void calculate_grid(double* A, double* Anew, size_t size) {
    // Параллельный расчет на GPU с использованием OpenACC
    #pragma acc parallel loop present(A, Anew)
    for (int i = 1; i < size-1; ++i) {
        #pragma acc loop
        for (int j = 1; j < size-1; ++j) {
            Anew[i*size + j] = 0.25 * (A[(i+1)*size + j] + A[(i-1)*size + j] + 
                                       A[i*size + j-1] + A[i*size + j+1]);
        }
    }
}

// Вычисление разницы между текущей и новой матрицей
void calculate_diff(double* A, double* Anew, double* diff, size_t size) {
    #pragma acc parallel loop present(A, Anew, diff)
    for (int i = 1; i < size-1; i++) {
        #pragma acc loop
        for (int j = 1; j < size-1; j++) {
            // Абсолютная разница для каждого элемента
            diff[(i-1) * (size-2) + (j-1)] = fabs(A[i * size + j] - Anew[i * size + j]);
        }
    }
}

// Копирование новых значений в основную матрицу
void copy_matrix(double* A, double* Anew, size_t size) {
    #pragma acc parallel loop present(A, Anew)
    for (int i = 1; i < size-1; i++) {
        #pragma acc loop
        for (int j = 1; j < size-1; j++) {
            A[i * size + j] = Anew[i * size + j];
        }
    }
}

// Освобождение памяти
void deallocate(double* A, double* Anew, double* diff, size_t size) {
    #pragma acc exit data delete(A[:size*size], Anew[:size*size], diff[:(size-2)*(size-2)])
    free(A);
    free(Anew);
    cudaFree(diff);
}


// Вывод сетки (только для маленьких размеров)
void print_grid(double* A, size_t size) {
    #pragma acc update self(A[:size*size])
    
    std::cout << "\nМатрица " << size << "x" << size << ":\n";
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cout << std::fixed << std::setprecision(4) << std::setw(8) << A[i*size + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    int size;
    double accuracy;
    int max_iterations;

    // Настройка обработки командной строки
    po::options_description desc("Опции");
    desc.add_options()
    ("size", po::value<int>(&size)->default_value(256))
    ("accuracy", po::value<double>(&accuracy)->default_value(1e-6))
    ("max_iterations", po::value<int>(&max_iterations)->default_value(1e+6));
    
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    
    std::cout << "Запуск программы (GPU версия с cuBLAS)!\n";
    std::cout << "Размер сетки: " << size << "x" << size << "\n";
    std::cout << "Точность: " << accuracy << "\n";
    std::cout << "Максимальное количество итераций: " << max_iterations << "\n\n";

    // Выделение памяти на хосте
    double* A = (double*)malloc(sizeof(double) * size * size);
    double* Anew = (double*)malloc(sizeof(double) * size * size);
    
    double* diff;
    int diff_size = (size-2) * (size-2);
    cudaMalloc((void**)&diff, sizeof(double) * diff_size);

    // Инициализация переменных итераций
    double error = accuracy + 1.0;
    int iteration = 0;
    int max_error_index = 0;
    
    // Создание обработчика cuBLAS с умным указателем для автоматического освобождения
    auto cublas_deleter = [](cublasHandle_t* handle) {
        if (handle && *handle) {
            cublasDestroy(*handle);
            delete handle;
        }
    };
    std::unique_ptr<cublasHandle_t, decltype(cublas_deleter)> handle(new cublasHandle_t, cublas_deleter);
    cublasCreate(handle.get());
    
    nvtxRangePushA("Initialization"); // Начало NVTX-диапазона для профилирования
    initialize(A, Anew, size);
    #pragma acc enter data copyin(A[:size*size], Anew[:size*size]) create(diff[:diff_size])
    nvtxRangePop();
    
    const auto start{std::chrono::steady_clock::now()};
    
    nvtxRangePushA("Main loop");
    while (error > accuracy && iteration < max_iterations) {
        // Расчет новых значений
        nvtxRangePushA("Calculate grid");
        calculate_grid(A, Anew, size);
        nvtxRangePop();
        
        // Периодическая проверка ошибки (каждые 1000 итераций)
        if (iteration % 1000 == 0) {
            nvtxRangePushA("Error calculation");
            // Вычисление разницы между итерациями
            calculate_diff(A, Anew, diff, size);
            
            // Использование cuBLAS для поиска максимальной ошибки
            #pragma acc host_data use_device(diff)
            {
                // Нахождение индекса максимального элемента
                cublasIdamax(*handle, diff_size, diff, 1, &max_error_index);
                // Копирование значения ошибки с GPU на CPU
                cudaMemcpy(&error, &diff[max_error_index-1], sizeof(double), cudaMemcpyDeviceToHost);
            }
            nvtxRangePop();
            
            if (iteration % 10000 == 0) {
                std::cout << "Итерация: " << iteration << ", ошибка: " << error << "\n";
            }
        }
        
        // Обновление матрицы
        nvtxRangePushA("Copy matrix");
        copy_matrix(A, Anew, size);
        nvtxRangePop();
        
        iteration++;
    }
    nvtxRangePop();
    
    // Замер времени выполнения
    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{end - start};

    std::cout << "\nРезультаты:\n";
    std::cout << "Время выполнения: " << elapsed_seconds.count() << " секунд\n";
    std::cout << "Количество итераций: " << iteration << "\n";
    std::cout << "Конечная ошибка: " << error << "\n";
    
    // Дополнительные тесты для маленьких сеток
    if (size == 10 || size == 13) {
        print_grid(A, size);
        deallocate(A, Anew, diff, size);
    } else {
        deallocate(A, Anew, diff, size);
        
        for (int test_size : {10, 13}) {
            size = test_size;
            diff_size = (size-2) * (size-2);
            
            A = (double*)malloc(size * size * sizeof(double));
            Anew = (double*)malloc(size * size * sizeof(double));
            cudaMalloc((void**)&diff, sizeof(double) * diff_size);
            
            initialize(A, Anew, size);
            #pragma acc enter data copyin(A[:size*size], Anew[:size*size]) create(diff[:diff_size])
            
            error = accuracy + 1.0;
            iteration = 0;
            
            while (error > accuracy && iteration < max_iterations) {
                calculate_grid(A, Anew, size);
                calculate_diff(A, Anew, diff, size);
                
                // Упрощенный цикл для маленьких сеток
                #pragma acc host_data use_device(diff)
                {
                    cublasIdamax(*handle, diff_size, diff, 1, &max_error_index);
                    cudaMemcpy(&error, &diff[max_error_index-1], sizeof(double), cudaMemcpyDeviceToHost);
                }
                
                copy_matrix(A, Anew, size);
                iteration++;
            }
            
            print_grid(A, size);
            deallocate(A, Anew, diff, size);
        }
    }
    return 0;
}