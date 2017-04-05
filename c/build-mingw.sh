#! /bin/bash

set -x

gcc *.c -c -Wall -fPIC -DNDEBUG -O3 -funroll-loops
gcc *.o -shared -o lbfgs.dll -Wl,--out-implib,lbfgs.lib -Wl,--version-script=lbfgs.export
