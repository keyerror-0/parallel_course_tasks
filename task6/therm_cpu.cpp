#include <iostream>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <boost/program_options.hpp>
#include <nvtx3/nvToolsExt.h>

namespace po = boost::program_options;

int main(int argc, char* argv[]) {
    int grid_size, max_iter;
    double target_accuracy;
    po::options_description opts("Program Options");
    opts.add_options()
        ("help", "display help")
        ("size", po::value<int>(&grid_size)->default_value(128), "grid dimension (N x N)")
        ("accuracy", po::value<double>(&target_accuracy)->default_value(1e-6), "error threshold")
        ("max_iterations", po::value<int>(&max_iter)->default_value(1000000), "iteration limit");
    po::variables_map vars;
    po::store(po::parse_command_line(argc, argv, opts), vars);
    po::notify(vars);
    if (vars.count("help")) {
        std::cout << opts << "\n";
        return 1;
    }
    std::cout << "Starting simulation...\n";
    double* grid = (double*)malloc(grid_size * grid_size * sizeof(double));
    double* grid_new = (double*)malloc(grid_size * grid_size * sizeof(double));
    nvtxRangePushA("initialize");
    for (size_t idx = 0; idx < grid_size * grid_size; ++idx) {
        grid[idx] = 0.0;
        grid_new[idx] = 0.0;
    }
    grid[0] = 10.0;
    grid[grid_size - 1] = 20.0;
    grid[grid_size * (grid_size - 1)] = 30.0;
    grid[grid_size * grid_size - 1] = 20.0;
    double tl = grid[0], tr = grid[grid_size - 1];
    double bl = grid[grid_size * (grid_size - 1)], br = grid[grid_size * grid_size - 1];
    for (int k = 1; k < grid_size - 1; ++k) {
        grid[k] = tl + (tr - tl) * k / (grid_size - 1.0);
        grid[grid_size * (grid_size - 1) + k] = bl + (br - bl) * k / (grid_size - 1.0);
        grid[grid_size * k] = tl + (bl - tl) * k / (grid_size - 1.0);
        grid[grid_size * k + grid_size - 1] = tr + (br - tr) * k / (grid_size - 1.0);
    }
    nvtxRangePop();
    double max_error = target_accuracy + 1.0;
    int current_iter = 0;
    auto start_time = std::chrono::steady_clock::now();
    nvtxRangePushA("computation");
    while (max_error > target_accuracy && current_iter < max_iter) {
        max_error = 0.0;
        #pragma acc parallel loop reduction(max:max_error)
        for (int row = 1; row < grid_size - 1; row++) {
            for (int col = 1; col < grid_size - 1; col++) {
                size_t idx = row * grid_size + col;
                grid_new[idx] = 0.25 * (
                    grid[(row + 1) * grid_size + col] +
                    grid[(row - 1) * grid_size + col] +
                    grid[row * grid_size + col - 1] +
                    grid[row * grid_size + col + 1]
                );
                max_error = std::fmax(max_error, std::fabs(grid_new[idx] - grid[idx]));
            }
        }
        #pragma acc parallel loop
        for (int row = 1; row < grid_size - 1; row++) {
            for (int col = 1; col < grid_size - 1; col++) {
                grid[row * grid_size + col] = grid_new[row * grid_size + col];
            }
        }
        current_iter++;
    }
    nvtxRangePop();
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    std::cout << "Execution time: " << duration.count() << " seconds\n";
    std::cout << "Total iterations: " << current_iter << "\n";
    std::cout << "Final error: " << max_error << "\n";
    free(grid);
    free(grid_new);
    return 0;
}