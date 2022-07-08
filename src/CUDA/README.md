# CUDA C Version - GPU general programming

## Compilation (Windows)

CWD: build/debug, so all the build files get put into the build dir

Note: nvcc uses an external compiler for CPU code, in the Windows use case it is cl.exe, bundled with VisualStudio.

Change the -I, -L and --compiler-bindir arguments to match your install directories

Debug:\
`nvcc.exe -g -lineinfo -arch=sm_61 -rdc=true ../../src/CUDA/main.cu ../../src/CUDA/nn_tools.cu ../../src/CUDA/getopt.cu -o cdl-cuda.exe "-ID:/Programs/NVIDIA GPU Computing Toolkit/CUDA/v11.7/include" -I../../src "-LD:/Programs/NVIDIA GPU Computing Toolkit/CUDA/v11.7/lib/x64" -lcudart_static -lcudadevrt -lcurand --compiler-bindir D:/Programs/Microsoft/VisualStudio/VC/Tools/MSVC/14.29.30133/bin/Hostx64/x64/cl.exe`