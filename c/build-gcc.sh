#! /bin/bash

set -x

gcc *.c -fPIC -DNDEBUG -c -I. -O3 -funroll-loops
gcc *.o -shared -o liblbfgs.so -Wl,--version-script=lbfgs.export

