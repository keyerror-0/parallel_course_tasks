all: cpu_one cpu_multi gpu

cpu_one:
	pgc++ -o therm_one -lboost_program_options -acc=host -Minfo=all -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include therm_cpu.cpp
	./therm_one --size=128 --accuracy=0.000001 --max_iterations=1000000

cpu_multi:
	pgc++ -o therm_multi -lboost_program_options -acc=multicore -Minfo=all -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include therm_cpu.cpp
	./therm_multi --size=128 --accuracy=0.000001 --max_iterations=1000000

gpu:
	pgc++ -o therm_gpu -lboost_program_options -acc=gpu -Minfo=all -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include therm_gpu.cpp
	./therm_gpu --size=128 --accuracy=0.000001 --max_iterations=1000000
