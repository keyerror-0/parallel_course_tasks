#include <iostream>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <boost/program_options.hpp>
#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

namespace po = boost::program_options;

// Ядро для расчета новых значений температур
__global__ void calculate_grid(double* A, double* Anew, int size) {
    // Расчет индексов потока в 2D сетке
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // Обработка только внутренних точек
    if (i > 0 && i < size-1 && j > 0 && j < size-1) {
        // Пятиточечный шаблон
        Anew[i*size + j] = 0.25 * (A[(i+1)*size + j] + A[(i-1)*size + j] +
                                   A[i*size + j+1] + A[i*size + j-1]);
    }
}

// Ядро для расчета максимальной ошибки с использованием CUB
__global__ void calculate_diff(double* A, double* Anew, double* block_max_errors, int size) {
    // Инициализация CUB BlockReduce для 256 потоков в блоке    
    using BlockReduce = cub::BlockReduce<double, 256>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    double local_max = 0.0;
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    // Каждый поток обрабатывает несколько элементов для балансировки нагрузки
    for (int idx = thread_id; idx < size * size; idx += total_threads) {
        int i = idx / size;
        int j = idx % size;
        if (i > 0 && i < size-1 && j > 0 && j < size-1) {
            double diff = fabs(A[idx] - Anew[idx]);
            if (diff > local_max) 
                local_max = diff;
        }
    }
    
    // Редукция внутри блока для поиска максимума
    double block_max = BlockReduce(temp_storage).Reduce(local_max, cub::Max());
    // Сохранение результата блока
    if (threadIdx.x == 0) 
        block_max_errors[blockIdx.x] = block_max;
}

// Ядро для копирования обновленных значений
__global__ void copy_matrix(double* A, const double* Anew, int size) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= 1 && i < size - 1 && j >= 1 && j < size - 1) {
        A[i * size + j] = Anew[i * size + j];
    }
}

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

    for (int i = 1; i < size-1; ++i) {
        A[i] = top_left + (top_right - top_left) * i / static_cast<double>(size-1);
        A[size*(size-1) + i] = bottom_left + (bottom_right - bottom_left) * i / static_cast<double>(size-1);
        A[size*i] = top_left + (bottom_left - top_left) * i / static_cast<double>(size-1);
        A[size*i + size-1] = top_right + (bottom_right - top_right) * i / static_cast<double>(size-1);
    }
}


// Вывод сетки (для малых размеров)
void print_grid(double* A, size_t size) {
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

    // Парсинг параметров командной строки
    po::options_description desc("Опции");
    desc.add_options()
        ("help", "показать описание опций")
        ("size", po::value<int>(&size)->default_value(256), "размер сетки (NxN)")
        ("accuracy", po::value<double>(&accuracy)->default_value(1e-6), "максимально допустимая ошибка")
        ("max_iterations", po::value<int>(&max_iterations)->default_value(1e+6), "максимальное количество итераций");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    
    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    std::cout << "Запуск программы (CUDA версия с CUB и CUDA Graph)!\n";
    std::cout << "Размер сетки: " << size << "x" << size << "\n";
    std::cout << "Точность: " << accuracy << "\n";
    std::cout << "Максимальное количество итераций: " << max_iterations << "\n\n";

    double error = accuracy + 1.0;
    int iteration = 0;

    // Выделение памяти на хосте
    size_t grid_size = size * size * sizeof(double);
    double* host_A = (double*)malloc(grid_size);
    double* host_Anew = (double*)malloc(grid_size);

    // Инициализация сетки
    nvtxRangePushA("Initialization");
    initialize(host_A, host_Anew, size);
    nvtxRangePop();


    // Выделение памяти на устройстве
    double* device_A, *device_Anew;
    cudaMalloc(&device_A, grid_size);
    cudaMalloc(&device_Anew, grid_size);
    cudaMemcpy(device_A, host_A, grid_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_Anew, host_Anew, grid_size, cudaMemcpyHostToDevice);

    // Настройка параметров запуска ядер
    int threads = 16;
    dim3 block(threads, threads);
    dim3 grid_dim((size + threads - 1) / threads, (size + threads - 1) / threads);

    // Выделение памяти для редукции ошибки
    int num_blocks = 1024;
    double* d_block_max;
    cudaMalloc(&d_block_max, sizeof(double) * num_blocks);

    // Создание CUDA Stream и Graph
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    
    // Захват 100 итераций в CUDA Graph
    nvtxRangePushA("CUDA Graph Creation");
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    for (int i = 0; i < 100; i++) {
        // Последовательные вызовы ядер
        calculate_grid<<<grid_dim, block, 0, stream>>>(device_A, device_Anew, size);
        copy_matrix<<<grid_dim, block, 0, stream>>>(device_A, device_Anew, size);
    }
    // Дополнительный вызов для расчета следующего состояния
    calculate_grid<<<grid_dim, block, 0, stream>>>(device_A, device_Anew, size);

    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
    nvtxRangePop();

    // Запуск основного цикла
    const auto start{std::chrono::steady_clock::now()};

    nvtxRangePushA("Main loop");
    while (error > accuracy && iteration < max_iterations) {
        // Запуск захваченного графа (100 итераций)
        nvtxRangePushA("CUDA Graph Launch");
        cudaGraphLaunch(instance, stream);
        cudaStreamSynchronize(stream);
        nvtxRangePop();

        // Расчет текущей ошибки
        nvtxRangePushA("Error calculation");
        calculate_diff<<<num_blocks, 256, 0, stream>>>(device_A, device_Anew, d_block_max, size);
        
        // Копирование и обработка результатов редукции
        double* h_block_max = new double[num_blocks];
        cudaMemcpy(h_block_max, d_block_max, sizeof(double) * num_blocks, cudaMemcpyDeviceToHost);

        error = 0.0;
        for (int i = 0; i < num_blocks; ++i) {
            if (h_block_max[i] > error) {
                error = h_block_max[i];
            }
        }

        delete[] h_block_max;
        nvtxRangePop();
        
        // Увеличение счетчика итераций
        iteration += 100;

        if (iteration % 10000 == 0) {
            std::cout << "Итерация: " << iteration << ", ошибка: " << error << "\n";
        }
    }
    nvtxRangePop();

    // Замер времени выполнения
    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{end - start};

    std::cout << "\nРезультаты:\n";
    std::cout << "Время выполнения: " << elapsed_seconds.count() << " секунд\n";
    std::cout << "Количество итераций: " << iteration << "\n";
    std::cout << "Конечная ошибка: " << error << std::endl;

    // Дополнительный вывод для малых сеток
    if (size <= 13) {
        cudaMemcpy(host_A, device_A, grid_size, cudaMemcpyDeviceToHost);
        print_grid(host_A, size);
    }

    // Освобождение ресурсов
    cudaFree(device_A);
    cudaFree(device_Anew);
    cudaFree(d_block_max);
    free(host_A);
    free(host_Anew);
    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(instance);
    cudaStreamDestroy(stream);

    return 0;
}


/*
Метрика	                OpenACC	    CUDA+CUB+Graph
Время (128×128)	        0.679594 с	0.10 с
Время (1024×1024)	    73.2928 с	60.9069с
Потребление памяти	    Высокое	    Оптимизированное
*/