#! /bin/bash

set -x

gcc *.c -c -Wall -fPIC -DNDEBUG -O3 -funroll-loops
gcc *.o -shared -o liblbfgs.so -lm -Wl,--no-undefined -Wl,--version-script=lbfgs.export
