# C Version - Standard CPU programming

## Compilation (Windows)

CWD: Project root

Release:\
`gcc -fdiagnostics-color=always src/C/main.c src/C/nn_tools.c src/C/getopt.c -Isrc -o build/release/cdl.exe -lm -O2 -Wall`

Debug:\
`gcc -fdiagnostics-color=always -g src/C/main.c src/C/nn_tools.c src/C/getopt.c -Isrc -o build/debug/cdl.exe -lm`