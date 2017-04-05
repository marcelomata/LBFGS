#! /bin/bash

set -x

gcc *.c -fPIC -DNDEBUG -c -I. -O3 -funroll-loops
gcc *.o -shared -o lbfgs.dll -Wl,--out-implib,lbfgs.lib -Wl,--version-script=lbfgs.export

