cuda = 	-I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include \
      	-I/usr/local/cuda/include \
      	-L/usr/local/cuda/lib64 \
      	-I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/math_libs/12.3/targets/x86_64-linux/include \
      	-L/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/math_libs/12.3/targets/x86_64-linux/lib \
      	-lcublas -lcudart -lnvToolsExt

gpu:
	pgc++ -o task -std=c++11 -lboost_program_options ${cuda} -acc=gpu -Minfo=all task.cpp
	./task --size=128 --accuracy=0.000001 --max_iterations=1000000

profile:
	nsys profile --trace=nvtx,cuda,openacc --stats=true ./task --size=256 --accuracy=0.0001 --max_iterations=50