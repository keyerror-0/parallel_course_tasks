gpu:
	nvcc -lboost_program_options -o 1 main.cu
	./1 --size=128 --accuracy=0.000001 --max_iterations=1000000

profile:
	nsys profile --trace=nvtx,cuda --stats=true ./1 --size=128 --accuracy=0.0001 --max_iterations=50