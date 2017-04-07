# LBFGS

---

A lightweight LBFGS optimizer for C/C++/Java.

This repository originates from [chokkan's implementation](https://github.com/chokkan/liblbfgs). This is probably the best LBFGS for its robust line search algorithm. 

However, the C code goes far away from its original version.

- It is lightweight: only two files.
- It is portable: it can be compiled with most C compilers and most platforms.
- It can leverage a highly optimized **CBLAS** library if exists.
    - Ugly manual SSE instructions are abandoned, because [GCC Auto Vectorization](https://gcc.gnu.org/projects/tree-ssa/vectorization.html) and loop unrolling techniques are supported now.

Furthermore, a 100% pure Java library **lbfgs4j** is re-written.

## Build

Compared to building it, I suggest contains source files(for both C and Java) into your project.

### Build with CBLAS

If you want to use CBLAS and cblas.h is available, compile lbfgs.c with:

    -DHAVE_CBLAS_H=1

or if you want to use CBLAS and cblas.h is unavailable, compile lbfgs.c with:

    -DHAVE_EXTERN_CBLAS=1
