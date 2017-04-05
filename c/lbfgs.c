/**
* Unconstrained Limited memory BFGS(L-BFGS).
*
* Forked from https://github.com/chokkan/liblbfgs
*
* The MIT License
*
* Copyright (c) 1990 Jorge Nocedal
* Copyright (c) 2007-2010 Naoaki Okazaki
* Copyright (c) 2014-2017 Yafei Zhang
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*/
#if !defined _XOPEN_SOURCE
#define _XOPEN_SOURCE 600
#endif
#if !defined _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "lbfgs.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <memory.h>

#if defined _MSC_VER
#define isnan(x) _isnan(x)
#define isinf(x) (!_finite(x))
#define inline __inline
#endif

#if HAVE_CBLAS_H == 1
#include <cblas.h>
#elif HAVE_EXTERN_CBLAS == 1
double cblas_ddot(const int N, const double* X, const int incX, const double* Y,
                  const int incY);
double cblas_dnrm2(const int N, const double* X, const int incX);
void cblas_dswap(const int N, double* X, const int incX, double* Y,
                 const int incY);
void cblas_dcopy(const int N, const double* X, const int incX, double* Y,
                 const int incY);
void cblas_daxpy(const int N, const double alpha, const double* X,
                 const int incX, double* Y, const int incY);
void cblas_dscal(const int N, const double alpha, double* X, const int incX);
#else
#define OFFSET(N, incX) ((incX) > 0 ? 0 : ((N)-1) * (-(incX)))

static double cblas_ddot(const int N, const double* X, const int incX,
                         const double* Y, const int incY) {
  double r = 0;
  int i;
  int ix = OFFSET(N, incX);
  int iy = OFFSET(N, incY);

  for (i = 0; i < N; i++) {
    r += X[ix] * Y[iy];
    ix += incX;
    iy += incY;
  }

  return r;
}

static double cblas_dnrm2(const int N, const double* X, const int incX) {
  double scale = 0.0;
  double ssq = 1.0;
  int i;
  int ix = 0;

  if (N <= 0 || incX <= 0) {
    return 0;
  } else if (N == 1) {
    return fabs(X[0]);
  }

  for (i = 0; i < N; i++) {
    const double x = X[ix];

    if (x != 0.0) {
      const double ax = fabs(x);

      if (scale < ax) {
        ssq = 1.0 + ssq * (scale / ax) * (scale / ax);
        scale = ax;
      } else {
        ssq += (ax / scale) * (ax / scale);
      }
    }

    ix += incX;
  }

  return scale * sqrt(ssq);
}

static void cblas_dswap(const int N, double* X, const int incX, double* Y,
                        const int incY) {
  int i;
  int ix = OFFSET(N, incX);
  int iy = OFFSET(N, incY);
  for (i = 0; i < N; i++) {
    const double tmp = X[ix];
    X[ix] = Y[iy];
    Y[iy] = tmp;
    ix += incX;
    iy += incY;
  }
}

static void cblas_dcopy(const int N, const double* X, const int incX, double* Y,
                        const int incY) {
  int i;
  int ix = OFFSET(N, incX);
  int iy = OFFSET(N, incY);

  for (i = 0; i < N; i++) {
    Y[iy] = X[ix];
    ix += incX;
    iy += incY;
  }
}

static void cblas_daxpy(const int N, const double alpha, const double* X,
                        const int incX, double* Y, const int incY) {
  int i;

  if (alpha == 0.0) {
    return;
  }

  if (incX == 1 && incY == 1) {
    const int m = N % 4;

    for (i = 0; i < m; i++) {
      Y[i] += alpha * X[i];
    }

    for (i = m; i + 3 < N; i += 4) {
      Y[i] += alpha * X[i];
      Y[i + 1] += alpha * X[i + 1];
      Y[i + 2] += alpha * X[i + 2];
      Y[i + 3] += alpha * X[i + 3];
    }
  } else {
    int ix = OFFSET(N, incX);
    int iy = OFFSET(N, incY);

    for (i = 0; i < N; i++) {
      Y[iy] += alpha * X[ix];
      ix += incX;
      iy += incY;
    }
  }
}

static void cblas_dscal(const int N, const double alpha, double* X,
                        const int incX) {
  int i;
  int ix = 0;

  if (incX <= 0) {
    return;
  }

  for (i = 0; i < N; i++) {
    X[ix] *= alpha;
    ix += incX;
  }
}
#undef OFFSET
#endif

/************************************************************************/
/* utilities */
/************************************************************************/
inline static double min2(double a, double b) { return a <= b ? a : b; }

inline static double max2(double a, double b) { return a >= b ? a : b; }

inline static double max3(double a, double b, double c) {
  return max2(max2(a, b), c);
}

inline static void* xalloc(size_t size) {
  void* m = calloc(size, 1);
  if (m == 0) {
    fprintf(stderr, "calloc %d bytes failed!\n", (int)size);
    exit(1);
  }
  return m;
}

/************************************************************************/
/* vector operations */
/************************************************************************/
#if defined _MSC_VER || defined __MINGW32__
double* vecalloc(int size) {
  size_t bytes = size * sizeof(double);
  double* m = (double*)_aligned_malloc(bytes, 16);
  if (m == 0) {
    fprintf(stderr, "_aligned_malloc %d bytes failed!\n", (int)bytes);
    exit(1);
  }
  memset(m, 0, bytes);
  return m;
}

void vecfree(double* vec) { _aligned_free(vec); }
#else
double* vecalloc(int size) {
  size_t bytes = size * sizeof(double);
  void* m;
  if (posix_memalign(&m, 16, bytes) != 0) {
    fprintf(stderr, "posix_memalign %d bytes failed!\n", (int)bytes);
    exit(1);
  }
  memset(m, 0, bytes);
  return (double*)m;
}

void vecfree(double* vec) { free(vec); }
#endif

void veccpy(double* y, const double* x, const int n) {
  cblas_dcopy(n, x, 1, y, 1);
}

void vecncpy(double* y, const double* x, const int n) {
  cblas_dcopy(n, x, 1, y, 1);
  cblas_dscal(n, -1.0, y, 1);
}

void vecadd(double* y, const double* x, const double c, const int n) {
  cblas_daxpy(n, c, x, 1, y, 1);
}

void vecdiff(double* z, const double* x, const double* y, const int n) {
  cblas_dcopy(n, x, 1, z, 1);
  cblas_daxpy(n, -1.0, y, 1, z, 1);
}

void vecscale(double* y, const double c, const int n) {
  cblas_dscal(n, c, y, 1);
}

void vecdot(double* s, const double* x, const double* y, const int n) {
  *s = cblas_ddot(n, x, 1, y, 1);
}

void vec2norm(double* s, const double* x, const int n) {
  *s = cblas_dnrm2(n, x, 1);
}

void vec2norminv(double* s, const double* x, const int n) {
  vec2norm(s, x, n);
  *s = (1.0 / *s);
}

void vecswap(double* x, double* y, const int n) { cblas_dswap(n, x, 1, y, 1); }

/************************************************************************/
/* L-BFGS */
/************************************************************************/
typedef struct {
  int n;
  void* instance;
  lbfgs_evaluate_t evaluate;
  lbfgs_progress_t progress;
} callback_data_t;

typedef struct {
  double alpha;
  double* s; /* [n] */
  double* y; /* [n] */
  double ys; /* vecdot(y, s) */
} iteration_data_t;

typedef int (*line_search_proc_t)(int n, double* x, double* f, double* g,
                                  double* s, double* step, const double* xp,
                                  const double* gp, double* wa,
                                  callback_data_t* cd,
                                  const lbfgs_parameter_t* param);

static int line_search_backtracking(int n, double* x, double* f, double* g,
                                    double* s, double* step, const double* xp,
                                    const double* gp, double* wa,
                                    callback_data_t* cd,
                                    const lbfgs_parameter_t* param);

static int line_search_backtracking_owlqn(int n, double* x, double* f,
                                          double* g, double* s, double* step,
                                          const double* xp, const double* gp,
                                          double* wp, callback_data_t* cd,
                                          const lbfgs_parameter_t* param);

static int line_search_morethuente(int n, double* x, double* f, double* g,
                                   double* s, double* step, const double* xp,
                                   const double* gp, double* wa,
                                   callback_data_t* cd,
                                   const lbfgs_parameter_t* param);

static int update_trial_interval(double* x, double* fx, double* dx, double* y,
                                 double* fy, double* dy, double* t, double* ft,
                                 double* dt, const double tmin,
                                 const double tmax, int* brackt);

static double owlqn_x1norm(const double* x, const int start, const int end);

static void owlqn_pseudo_gradient(double* pg, const double* x, const double* g,
                                  const int n, const double c, const int start,
                                  const int end);

static void owlqn_project(double* d, const double* sign, const int start,
                          const int end);

static void owlqn_contrain_line_search(double* d, const double* pg,
                                       const int start, const int end);

static const lbfgs_parameter_t default_param = {
    6,       1e-5,  0,    1e-5, 0,   LBFGS_LINESEARCH_DEFAULT,
    40,      1e-20, 1e20, 1e-4, 0.9, 0.9,
    1.0e-16, 0.0,   0,    -1};

void lbfgs_default_parameter(lbfgs_parameter_t* param) {
  memcpy(param, &default_param, sizeof(*param));
}

static int default_lbfgs_progress(void* instance, int n, const double* x,
                                  const double* g, const double fx,
                                  const double xnorm, const double gnorm,
                                  const double step, int k, int n_evaluate) {
  printf("Iteration %d:\n", k);
  printf("    fx=%8.8lf, evaluations of fx=%d\n", fx, n_evaluate);
  printf("    xnorm=%8.8lf, gnorm=%8.8lf, step=%8.8lf\n", xnorm, gnorm, step);
  printf("\n");
  return 0;
}

int lbfgs(int n, double* x, double* pfx, lbfgs_evaluate_t evaluate,
          lbfgs_progress_t progress, void* instance,
          const lbfgs_parameter_t* _param) {
  int ret;
  int i, j, k, ls, end, bound, n_evaluate = 0;
  int enalbe_owlqn;
  double step;
  lbfgs_parameter_t param = (_param) ? (*_param) : default_param;
  const int m = param.m;
  double* xp;
  double *g, *gp, *pg = 0;
  double *d, *w, *pf = 0;
  iteration_data_t *lm = 0, *it = 0;
  double ys, yy;
  double xnorm, gnorm, rate, beta;
  double fx;
  line_search_proc_t linesearch = line_search_morethuente;

  callback_data_t cd;
  cd.n = n;
  cd.instance = instance;
  cd.evaluate = evaluate;
  cd.progress = (progress) ? progress : default_lbfgs_progress;

  /* Check the input parameters for errors. */
  if (n <= 0) {
    return LBFGSERR_INVALID_N;
  }
  if (param.epsilon < 0.0) {
    return LBFGSERR_INVALID_EPSILON;
  }
  if (param.past < 0) {
    return LBFGSERR_INVALID_TESTPERIOD;
  }
  if (param.delta < 0.0) {
    return LBFGSERR_INVALID_DELTA;
  }
  if (param.min_step < 0.0) {
    return LBFGSERR_INVALID_MINSTEP;
  }
  if (param.max_step < param.min_step) {
    return LBFGSERR_INVALID_MAXSTEP;
  }
  if (param.ftol < 0.0) {
    return LBFGSERR_INVALID_FTOL;
  }
  if (param.linesearch == LBFGS_LINESEARCH_BACKTRACKING_WOLFE ||
      param.linesearch == LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE) {
    if (param.wolfe <= param.ftol || 1. <= param.wolfe) {
      return LBFGSERR_INVALID_WOLFE;
    }
  }
  if (param.gtol < 0.0) {
    return LBFGSERR_INVALID_GTOL;
  }
  if (param.xtol < 0.0) {
    return LBFGSERR_INVALID_XTOL;
  }
  if (param.max_linesearch <= 0) {
    return LBFGSERR_INVALID_MAXLINESEARCH;
  }
  if (param.orthantwise_c < 0.0) {
    return LBFGSERR_INVALID_ORTHANTWISE;
  }
  if (param.orthantwise_start < 0 || param.orthantwise_start > n) {
    return LBFGSERR_INVALID_ORTHANTWISE_START;
  }
  if (param.orthantwise_end < 0) {
    param.orthantwise_end = n;
  }
  if (param.orthantwise_end > n) {
    return LBFGSERR_INVALID_ORTHANTWISE_END;
  }

  enalbe_owlqn = (param.orthantwise_c != 0.0);
  if (enalbe_owlqn) {
    switch (param.linesearch) {
      case LBFGS_LINESEARCH_BACKTRACKING_WOLFE:
        linesearch = line_search_backtracking_owlqn;
        break;
      default:
        /* Only the backtracking method is available. */
        return LBFGSERR_INVALID_LINESEARCH;
    }
  } else {
    switch (param.linesearch) {
      case LBFGS_LINESEARCH_MORETHUENTE:
        linesearch = line_search_morethuente;
        break;
      case LBFGS_LINESEARCH_BACKTRACKING_ARMIJO:
      case LBFGS_LINESEARCH_BACKTRACKING_WOLFE:
      case LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE:
        linesearch = line_search_backtracking;
        break;
      default:
        return LBFGSERR_INVALID_LINESEARCH;
    }
  }

  /* Allocate working space. */
  xp = vecalloc(n);
  g = vecalloc(n);
  gp = vecalloc(n);
  d = vecalloc(n);
  w = vecalloc(n);

  /* Allocate pseudo gradient. */
  if (enalbe_owlqn) {
    pg = vecalloc(n);
  }

  /* Allocate and initialize the limited memory storage. */
  lm = (iteration_data_t*)xalloc(m * sizeof(iteration_data_t));
  for (i = 0; i < m; i++) {
    it = &lm[i];
    it->alpha = 0.0;
    it->s = vecalloc(n);
    it->y = vecalloc(n);
    it->ys = 0.0;
  }

  /* Allocate an array for storing previous values of the objective function. */
  if (param.past > 0) {
    pf = vecalloc(param.past);
  }

  fx = cd.evaluate(cd.instance, cd.n, x, g, 0);
  n_evaluate++;

  if (enalbe_owlqn) {
    xnorm = owlqn_x1norm(x, param.orthantwise_start, param.orthantwise_end);
    fx += xnorm * param.orthantwise_c;
    owlqn_pseudo_gradient(pg, x, g, n, param.orthantwise_c,
                          param.orthantwise_start, param.orthantwise_end);
  }

  /* Store the initial value of the objective function. */
  if (pf) {
    pf[0] = fx;
  }

  /**
  * Compute the direction.
  * we assume the initial hessian matrix H_0 as the identity matrix.
  */
  if (!enalbe_owlqn) {
    vecncpy(d, g, n);
  } else {
    vecncpy(d, pg, n);
  }

  /**
  * Make sure that the initial variables are not a minimizer.
  */
  vec2norm(&xnorm, x, n);
  if (!enalbe_owlqn) {
    vec2norm(&gnorm, g, n);
  } else {
    vec2norm(&gnorm, pg, n);
  }
  if (xnorm < 1.0) {
    xnorm = 1.0;
  }
  if (gnorm / xnorm <= param.epsilon) {
    ret = LBFGS_ALREADY_MINIMIZED;
    goto lbfgs_exit;
  }

  /**
  * Compute the initial step:
  * step = 1.0 / ||d||
  */
  vec2norminv(&step, d, n);

  k = 1;
  end = 0;
  for (;;) {
    /* Store the current position and gradient vectors. */
    veccpy(xp, x, n);
    veccpy(gp, g, n);

    /* Search for an optimal step. */
    if (!enalbe_owlqn) {
      ls = linesearch(n, x, &fx, g, d, &step, xp, gp, w, &cd, &param);
    } else {
      ls = linesearch(n, x, &fx, g, d, &step, xp, pg, w, &cd, &param);
      owlqn_pseudo_gradient(pg, x, g, n, param.orthantwise_c,
                            param.orthantwise_start, param.orthantwise_end);
    }

    if (ls < 0) {
      /* Revert to the previous point. */
      veccpy(x, xp, n);
      veccpy(g, gp, n);
      ret = ls;
      break;
    }

    n_evaluate += ls;

    /* Compute x and g norms. */
    vec2norm(&xnorm, x, n);
    if (!enalbe_owlqn) {
      vec2norm(&gnorm, g, n);
    } else {
      vec2norm(&gnorm, pg, n);
    }

    /* Report the progress. */
    if ((ret = cd.progress(cd.instance, cd.n, x, g, fx, xnorm, gnorm, step, k,
                           n_evaluate)) != 0) {
      ret = LBFGSERR_CANCELED;
      break;
    }

    /* Convergence test. */
    if (xnorm < 1.0) {
      xnorm = 1.0;
    }
    if (gnorm / xnorm <= param.epsilon) {
      ret = LBFGS_CONVERGENCE;
      break;
    }

    /* Stopping criterion test. */
    if (pf) {
      /* We don't test the stopping criterion while k < past. */
      if (param.past <= k) {
        /* Compute the relative improvement from the past. */
        rate = (pf[k % param.past] - fx) / fx;

        /* The stopping criterion. */
        if (rate < param.delta) {
          ret = LBFGS_CONVERGENCE_DELTA;
          break;
        }
      }

      /* Store the current value of the objective function. */
      pf[k % param.past] = fx;
    }

    if (param.max_iterations != 0 && param.max_iterations < k + 1) {
      ret = LBFGSERR_MAXIMUMITERATION;
      break;
    }

    /**
    * Update s and y:
    * s_{k+1} = x_{k+1} - x_{k} = step * d_{k}
    * y_{k+1} = g_{k+1} - g_{k}
    */
    it = &lm[end];
    vecdiff(it->s, x, xp, n);
    vecdiff(it->y, g, gp, n);

    /**
    * Compute scalars ys and yy:
    * ys = y^t s = 1 / \rho
    * yy = y^t y
    * Notice that yy is used for scaling the hessian matrix H_0 (Cholesky
    * factor).
    */
    vecdot(&ys, it->y, it->s, n);
    vecdot(&yy, it->y, it->y, n);
    it->ys = ys;

    /**
    * Recursive formula to compute d = -(H g).
    * This is described in page 779 of:
    * Jorge Nocedal.
    * Updating Quasi-Newton Matrices with Limited Storage.
    * Mathematics of Computation, Vol. 35, No. 151,
    * pp. 773--782, 1980.
    */
    bound = (m <= k) ? m : k;
    k++;
    end = (end + 1) % m;

    /* Compute the steepest direction. */
    /* Compute the negative of (pseudo) gradient. */
    if (!enalbe_owlqn) {
      vecncpy(d, g, n);
    } else {
      vecncpy(d, pg, n);
    }

    j = end;
    for (i = 0; i < bound; i++) {
      j = (j + m - 1) % m; /* if (--j == -1) j = m-1; */
      it = &lm[j];
      /* \alpha_{j} = \rho_{j} s^{t}_{j} q_{k+1} */
      vecdot(&it->alpha, it->s, d, n);
      it->alpha /= it->ys;
      /* q_{i} = q_{i+1} - \alpha_{i} y_{i} */
      vecadd(d, it->y, -it->alpha, n);
    }

    vecscale(d, ys / yy, n);

    for (i = 0; i < bound; i++) {
      it = &lm[j];
      /* \beta_{j} = \rho_{j} y^t_{j} \gamma_{i} */
      vecdot(&beta, it->y, d, n);
      beta /= it->ys;
      /* \gamma_{i+1} = \gamma_{i} + (\alpha_{j} - \beta_{j}) s_{j} */
      vecadd(d, it->s, it->alpha - beta, n);
      j = (j + 1) % m; /* if (++j == m) j = 0; */
    }

    /* Constrain the search direction for orthant-wise updates. */
    if (enalbe_owlqn) {
      owlqn_contrain_line_search(d, pg, param.orthantwise_start,
                                 param.orthantwise_end);
    }

    /* Now the search direction d is ready. We try step = 1 first. */
    step = 1.0;
  }

lbfgs_exit:
  /* Return the final value of the objective function. */
  if (pfx) {
    *pfx = fx;
  }

  vecfree(pf);
  if (lm != 0) {
    for (i = 0; i < m; i++) {
      vecfree(lm[i].s);
      vecfree(lm[i].y);
    }
    free(lm);
  }
  vecfree(pg);
  vecfree(w);
  vecfree(d);
  vecfree(gp);
  vecfree(g);
  vecfree(xp);
  return ret;
}

static int line_search_backtracking(int n, double* x, double* f, double* g,
                                    double* s, double* step, const double* xp,
                                    const double* gp, double* wp,
                                    callback_data_t* cd,
                                    const lbfgs_parameter_t* param) {
  int count = 0;
  double width, dg;
  double finit, dginit = 0.0, dgtest;
  const double dec = 0.5, inc = 2.1;

  /* Check the input parameters for errors. */
  if (*step <= 0.0) {
    return LBFGSERR_INVALIDPARAMETERS;
  }

  /* Compute the initial gradient in the search direction. */
  vecdot(&dginit, g, s, n);

  /* Make sure that s points to a descent direction. */
  if (0 < dginit) {
    return LBFGSERR_INCREASEGRADIENT;
  }

  /* The initial value of the objective function. */
  finit = *f;
  dgtest = param->ftol * dginit;

  for (;;) {
    veccpy(x, xp, n);
    vecadd(x, s, *step, n);

    *f = cd->evaluate(cd->instance, cd->n, x, g, *step);
    count++;

    if (*f > finit + *step * dgtest) {
      width = dec;
    } else {
      /* The sufficient decrease condition (Armijo condition). */
      if (param->linesearch == LBFGS_LINESEARCH_BACKTRACKING_ARMIJO) {
        /* Exit with the Armijo condition. */
        return count;
      }

      /* Check the Wolfe condition. */
      vecdot(&dg, g, s, n);
      if (dg < param->wolfe * dginit) {
        width = inc;
      } else {
        if (param->linesearch == LBFGS_LINESEARCH_BACKTRACKING_WOLFE) {
          /* Exit with the regular Wolfe condition. */
          return count;
        }

        /* Check the strong Wolfe condition. */
        if (dg > -param->wolfe * dginit) {
          width = dec;
        } else {
          /* Exit with the strong Wolfe condition. */
          return count;
        }
      }
    }

    if (*step < param->min_step) {
      /* The step is the minimum value. */
      return LBFGSERR_MINIMUMSTEP;
    }
    if (*step > param->max_step) {
      /* The step is the maximum value. */
      return LBFGSERR_MAXIMUMSTEP;
    }
    if (count >= param->max_linesearch) {
      /* Maximum number of iteration. */
      return LBFGSERR_MAXIMUMLINESEARCH;
    }

    (*step) *= width;
  }
}

static int line_search_backtracking_owlqn(int n, double* x, double* f,
                                          double* g, double* s, double* step,
                                          const double* xp, const double* gp,
                                          double* wp, callback_data_t* cd,
                                          const lbfgs_parameter_t* param) {
  int i, count = 0;
  double width = 0.5, norm = 0.0;
  double finit = *f, dgtest;

  /* Check the input parameters for errors. */
  if (*step <= 0.0) {
    return LBFGSERR_INVALIDPARAMETERS;
  }

  /* Choose the orthant for the new point. */
  for (i = 0; i < n; i++) {
    wp[i] = (xp[i] == 0.0) ? -gp[i] : xp[i];
  }

  for (;;) {
    veccpy(x, xp, n);
    vecadd(x, s, *step, n);

    /* The current point is projected onto the orthant. */
    owlqn_project(x, wp, param->orthantwise_start, param->orthantwise_end);

    *f = cd->evaluate(cd->instance, cd->n, x, g, *step);
    count++;

    /* Compute the L1 norm of the variables and add it to the object value. */
    norm = owlqn_x1norm(x, param->orthantwise_start, param->orthantwise_end);
    *f += norm * param->orthantwise_c;

    dgtest = 0.0;
    for (i = 0; i < n; i++) {
      dgtest += (x[i] - xp[i]) * gp[i];
    }

    if (*f <= finit + param->ftol * dgtest) {
      /* The sufficient decrease condition. */
      return count;
    }

    if (*step < param->min_step) {
      /* The step is the minimum value. */
      return LBFGSERR_MINIMUMSTEP;
    }
    if (*step > param->max_step) {
      /* The step is the maximum value. */
      return LBFGSERR_MAXIMUMSTEP;
    }
    if (count >= param->max_linesearch) {
      /* Maximum number of iteration. */
      return LBFGSERR_MAXIMUMLINESEARCH;
    }

    (*step) *= width;
  }
}

/**
* return 0, if sign(x) = sign(y)
* return 1, if sign(x) != sign(y)
*/
inline static int fsigndiff(double x, double y) {
  return x * (y / fabs(y)) < 0.0;
}

static int line_search_morethuente(int n, double* x, double* f, double* g,
                                   double* s, double* step, const double* xp,
                                   const double* gp, double* wa,
                                   callback_data_t* cd,
                                   const lbfgs_parameter_t* param) {
  int count = 0;
  int brackt, stage1, uinfo = 0;
  double dg;
  double stx, fx, dgx;
  double sty, fy, dgy;
  double fxm, dgxm, fym, dgym, fm, dgm;
  double finit, ftest1, dginit, dgtest;
  double width, prev_width;
  double stmin, stmax;

  /* Check the input parameters for errors. */
  if (*step <= 0.0) {
    return LBFGSERR_INVALIDPARAMETERS;
  }

  /* Compute the initial gradient in the search direction. */
  vecdot(&dginit, g, s, n);

  /* Make sure that s points to a descent direction. */
  if (0 < dginit) {
    return LBFGSERR_INCREASEGRADIENT;
  }

  /* Initialize local variables. */
  brackt = 0;
  stage1 = 1;
  finit = *f;
  dgtest = param->ftol * dginit;
  width = param->max_step - param->min_step;
  prev_width = 2.0 * width;

  /**
  * The variables stx, fx, dgx contain the values of the step,
  * function, and directional derivative at the best step.
  * The variables sty, fy, dgy contain the value of the step,
  * function, and derivative at the other endpoint of
  * the interval of uncertainty.
  * The variables step, f, dg contain the values of the step,
  * function, and derivative at the current step.
  */
  stx = sty = 0.0;
  fx = fy = finit;
  dgx = dgy = dginit;

  for (;;) {
    /**
    * Set the minimum and maximum steps to correspond to the
    * present interval of uncertainty.
    */
    if (brackt) {
      stmin = min2(stx, sty);
      stmax = max2(stx, sty);
    } else {
      stmin = stx;
      stmax = *step + 4.0 * (*step - stx);
    }

    /* Clip the step in the range of [stpmin, stpmax]. */
    if (*step < param->min_step) {
      *step = param->min_step;
    }
    if (param->max_step < *step) {
      *step = param->max_step;
    }

    /**
    * If an unusual termination is to occur then let
    * step be the lowest point obtained so far.
    */
    if ((brackt && ((*step <= stmin || stmax <= *step) ||
                    param->max_linesearch <= count + 1 || uinfo != 0)) ||
        (brackt && (stmax - stmin <= param->xtol * stmax))) {
      *step = stx;
    }

    veccpy(x, xp, n);
    vecadd(x, s, *step, n);

    *f = cd->evaluate(cd->instance, cd->n, x, g, *step);
    count++;

    vecdot(&dg, g, s, n);

    ftest1 = finit + *step * dgtest;

    /* Test for errors and convergence. */
    if (brackt && ((*step <= stmin || stmax <= *step) || uinfo != 0)) {
      /* Rounding errors prevent further progress. */
      return LBFGSERR_ROUNDING_ERROR;
    }
    if (*step == param->max_step && *f <= ftest1 && dg <= dgtest) {
      /* The step is the maximum value. */
      return LBFGSERR_MAXIMUMSTEP;
    }
    if (*step == param->min_step && (ftest1 < *f || dgtest <= dg)) {
      /* The step is the minimum value. */
      return LBFGSERR_MINIMUMSTEP;
    }
    if (brackt && (stmax - stmin) <= param->xtol * stmax) {
      /* Relative width of the interval of uncertainty is at most xtol. */
      return LBFGSERR_WIDTHTOOSMALL;
    }
    if (count >= param->max_linesearch) {
      /* Maximum number of iteration. */
      return LBFGSERR_MAXIMUMLINESEARCH;
    }
    if (*f <= ftest1 && fabs(dg) <= param->gtol * (-dginit)) {
      /* The sufficient decrease condition and the directional derivative
       * condition hold. */
      return count;
    }

    /**
    * In the first stage we seek a step for which the modified
    * function has a non-positive value and nonnegative derivative.
    */
    if (stage1 && *f <= ftest1 &&
        min2(param->ftol, param->gtol) * dginit <= dg) {
      stage1 = 0;
    }

    /**
    * A modified function is used to predict the step only if
    * we have not obtained a step for which the modified
    * function has a non-positive function value and nonnegative
    * derivative, and if a lower function value has been
    * obtained but the decrease is not sufficient.
    */
    if (stage1 && ftest1 < *f && *f <= fx) {
      /* Define the modified function and derivative values. */
      fm = *f - *step * dgtest;
      fxm = fx - stx * dgtest;
      fym = fy - sty * dgtest;
      dgm = dg - dgtest;
      dgxm = dgx - dgtest;
      dgym = dgy - dgtest;

      /**
      * Call update_trial_interval() to update the interval of
      * uncertainty and to compute the new step.
      */
      uinfo = update_trial_interval(&stx, &fxm, &dgxm, &sty, &fym, &dgym, step,
                                    &fm, &dgm, stmin, stmax, &brackt);

      /* Reset the function and gradient values for f. */
      fx = fxm + stx * dgtest;
      fy = fym + sty * dgtest;
      dgx = dgxm + dgtest;
      dgy = dgym + dgtest;
    } else {
      /**
      * Call update_trial_interval() to update the interval of
      * uncertainty and to compute the new step.
      */
      uinfo = update_trial_interval(&stx, &fx, &dgx, &sty, &fy, &dgy, step, f,
                                    &dg, stmin, stmax, &brackt);
    }

    /**
    * Force a sufficient decrease in the interval of uncertainty.
    */
    if (brackt) {
      if (0.66 * prev_width <= fabs(sty - stx)) {
        *step = stx + 0.5 * (sty - stx);
      }
      prev_width = width;
      width = fabs(sty - stx);
    }
  }

  return LBFGSERR_LOGICERROR;
}

/**
* Define the local variables for computing minimizers.
*/
#define USES_MINIMIZER double a, d, gamma, theta, p, q, r, s;

/**
* Find a minimizer of an interpolated cubic function.
*  @param  cm      The minimizer of the interpolated cubic.
*  @param  u       The value of one point, u.
*  @param  fu      The value of f(u).
*  @param  du      The value of f'(u).
*  @param  v       The value of another point, v.
*  @param  fv      The value of f(v).
*  @param  du      The value of f'(v).
*/
#define CUBIC_MINIMIZER(cm, u, fu, du, v, fv, dv)      \
  d = (v) - (u);                                       \
  theta = ((fu) - (fv)) * 3 / d + (du) + (dv);         \
  p = fabs(theta);                                     \
  q = fabs(du);                                        \
  r = fabs(dv);                                        \
  s = max3(p, q, r);                                   \
  /* gamma = s*sqrt((theta/s)**2 - (du/s) * (dv/s)) */ \
  a = theta / s;                                       \
  gamma = s * sqrt(a * a - ((du) / s) * ((dv) / s));   \
  if ((v) < (u)) gamma = -gamma;                       \
  p = gamma - (du) + theta;                            \
  q = gamma - (du) + gamma + (dv);                     \
  r = p / q;                                           \
  (cm) = (u) + r * d;

/**
* Find a minimizer of an interpolated cubic function.
*  @param  cm      The minimizer of the interpolated cubic.
*  @param  u       The value of one point, u.
*  @param  fu      The value of f(u).
*  @param  du      The value of f'(u).
*  @param  v       The value of another point, v.
*  @param  fv      The value of f(v).
*  @param  du      The value of f'(v).
*  @param  xmin    The maximum value.
*  @param  xmin    The minimum value.
*/
#define CUBIC_MINIMIZER2(cm, u, fu, du, v, fv, dv, xmin, xmax) \
  d = (v) - (u);                                               \
  theta = ((fu) - (fv)) * 3 / d + (du) + (dv);                 \
  p = fabs(theta);                                             \
  q = fabs(du);                                                \
  r = fabs(dv);                                                \
  s = max3(p, q, r);                                           \
  /* gamma = s*sqrt((theta/s)**2 - (du/s) * (dv/s)) */         \
  a = theta / s;                                               \
  gamma = s * sqrt(max2(0, a * a - ((du) / s) * ((dv) / s)));  \
  if ((u) < (v)) gamma = -gamma;                               \
  p = gamma - (dv) + theta;                                    \
  q = gamma - (dv) + gamma + (du);                             \
  r = p / q;                                                   \
  if (r < 0. && gamma != 0.0) {                                \
    (cm) = (v)-r * d;                                          \
  } else if (a < 0) {                                          \
    (cm) = (xmax);                                             \
  } else {                                                     \
    (cm) = (xmin);                                             \
  }

/**
* Find a minimizer of an interpolated quadratic function.
*  @param  qm      The minimizer of the interpolated quadratic.
*  @param  u       The value of one point, u.
*  @param  fu      The value of f(u).
*  @param  du      The value of f'(u).
*  @param  v       The value of another point, v.
*  @param  fv      The value of f(v).
*/
#define QUARD_MINIMIZER(qm, u, fu, du, v, fv) \
  a = (v) - (u);                              \
  (qm) = (u) + (du) / (((fu) - (fv)) / a + (du)) / 2 * a;

/**
* Find a minimizer of an interpolated quadratic function.
*  @param  qm      The minimizer of the interpolated quadratic.
*  @param  u       The value of one point, u.
*  @param  du      The value of f'(u).
*  @param  v       The value of another point, v.
*  @param  dv      The value of f'(v).
*/
#define QUARD_MINIMIZER2(qm, u, du, v, dv) \
  a = (u) - (v);                           \
  (qm) = (v) + (dv) / ((dv) - (du)) * a;

/**
* Update a safeguarded trial value and interval for line search.
*
*  The parameter x represents the step with the least function value.
*  The parameter t represents the current step. This function assumes
*  that the derivative at the point of x in the direction of the step.
*  If the bracket is set to true, the minimizer has been bracketed in
*  an interval of uncertainty with endpoints between x and y.
*
*  @param  x       The pointer to the value of one endpoint.
*  @param  fx      The pointer to the value of f(x).
*  @param  dx      The pointer to the value of f'(x).
*  @param  y       The pointer to the value of another endpoint.
*  @param  fy      The pointer to the value of f(y).
*  @param  dy      The pointer to the value of f'(y).
*  @param  t       The pointer to the value of the trial value, t.
*  @param  ft      The pointer to the value of f(t).
*  @param  dt      The pointer to the value of f'(t).
*  @param  tmin    The minimum value for the trial value, t.
*  @param  tmax    The maximum value for the trial value, t.
*  @param  brackt  The pointer to the predicate if the trial value is
*                  bracketed.
*  @retval int     Status value. Zero indicates a normal termination.
*
*  @see
*      Jorge J. More and David J. Thuente. Line search algorithm with
*      guaranteed sufficient decrease. ACM Transactions on Mathematical
*      Software (TOMS), Vol 20, No 3, pp. 286-307, 1994.
*/
static int update_trial_interval(double* x, double* fx, double* dx, double* y,
                                 double* fy, double* dy, double* t, double* ft,
                                 double* dt, const double tmin,
                                 const double tmax, int* brackt) {
  int bound;
  int dsign = fsigndiff(*dt, *dx);
  double mc;      /* minimizer of an interpolated cubic. */
  double mq;      /* minimizer of an interpolated quadratic. */
  double newt;    /* new trial value. */
  USES_MINIMIZER; /* for CUBIC_MINIMIZER and QUARD_MINIMIZER. */

  /* Check the input parameters for errors. */
  if (*brackt) {
    if (*t <= min2(*x, *y) || max2(*x, *y) <= *t) {
      /* The trivial value t is out of the interval. */
      return LBFGSERR_OUTOFINTERVAL;
    }
    if (0.0 <= *dx * (*t - *x)) {
      /* The function must decrease from x. */
      return LBFGSERR_INCREASEGRADIENT;
    }
    if (tmax < tmin) {
      /* Incorrect tmin and tmax specified. */
      return LBFGSERR_INCORRECT_TMINMAX;
    }
  }

  /**
  * Trial value selection.
  */
  if (*fx < *ft) {
    /**
    * Case 1: a higher function value.
    * The minimum is brackt. If the cubic minimizer is closer
    * to x than the quadratic one, the cubic one is taken, else
    * the average of the minimizers is taken.
    */
    *brackt = 1;
    bound = 1;
    CUBIC_MINIMIZER(mc, *x, *fx, *dx, *t, *ft, *dt);
    QUARD_MINIMIZER(mq, *x, *fx, *dx, *t, *ft);
    if (fabs(mc - *x) < fabs(mq - *x)) {
      newt = mc;
    } else {
      newt = mc + 0.5 * (mq - mc);
    }
  } else if (dsign) {
    /**
    * Case 2: a lower function value and derivatives of
    * opposite sign. The minimum is brackt. If the cubic
    * minimizer is closer to x than the quadratic (secant) one,
    * the cubic one is taken, else the quadratic one is taken.
    */
    *brackt = 1;
    bound = 0;
    CUBIC_MINIMIZER(mc, *x, *fx, *dx, *t, *ft, *dt);
    QUARD_MINIMIZER2(mq, *x, *dx, *t, *dt);
    if (fabs(mc - *t) > fabs(mq - *t)) {
      newt = mc;
    } else {
      newt = mq;
    }
  } else if (fabs(*dt) < fabs(*dx)) {
    /**
    * Case 3: a lower function value, derivatives of the
    * same sign, and the magnitude of the derivative decreases.
    * The cubic minimizer is only used if the cubic tends to
    * infinity in the direction of the minimizer or if the minimum
    * of the cubic is beyond t. Otherwise the cubic minimizer is
    * defined to be either tmin or tmax. The quadratic (secant)
    * minimizer is also computed and if the minimum is brackt
    * then the the minimizer closest to x is taken, else the one
    * farthest away is taken.
    */
    bound = 1;
    CUBIC_MINIMIZER2(mc, *x, *fx, *dx, *t, *ft, *dt, tmin, tmax);
    QUARD_MINIMIZER2(mq, *x, *dx, *t, *dt);
    if (*brackt) {
      if (fabs(*t - mc) < fabs(*t - mq)) {
        newt = mc;
      } else {
        newt = mq;
      }
    } else {
      if (fabs(*t - mc) > fabs(*t - mq)) {
        newt = mc;
      } else {
        newt = mq;
      }
    }
  } else {
    /**
    * Case 4: a lower function value, derivatives of the
    * same sign, and the magnitude of the derivative does
    * not decrease. If the minimum is not brackt, the step
    * is either tmin or tmax, else the cubic minimizer is taken.
    */
    bound = 0;
    if (*brackt) {
      CUBIC_MINIMIZER(newt, *t, *ft, *dt, *y, *fy, *dy);
    } else if (*x < *t) {
      newt = tmax;
    } else {
      newt = tmin;
    }
  }

  /**
  * Update the interval of uncertainty. This update does not
  * depend on the new step or the case analysis above.
  * - Case a: if f(x) < f(t),
  * x <- x, y <- t.
  * - Case b: if f(t) <= f(x) && f'(t)*f'(x) > 0,
  * x <- t, y <- y.
  * - Case c: if f(t) <= f(x) && f'(t)*f'(x) < 0,
  * x <- t, y <- x.
  */
  if (*fx < *ft) {
    /* Case a */
    *y = *t;
    *fy = *ft;
    *dy = *dt;
  } else {
    /* Case c */
    if (dsign) {
      *y = *x;
      *fy = *fx;
      *dy = *dx;
    }
    /* Cases b and c */
    *x = *t;
    *fx = *ft;
    *dx = *dt;
  }

  /* Clip the new trial value in [tmin, tmax]. */
  if (tmax < newt) {
    newt = tmax;
  }
  if (newt < tmin) {
    newt = tmin;
  }

  /**
  * Redefine the new trial value if it is close to the upper bound
  * of the interval.
  */
  if (*brackt && bound) {
    mq = *x + 0.66 * (*y - *x);
    if (*x < *y) {
      if (mq < newt) {
        newt = mq;
      }
    } else {
      if (newt < mq) {
        newt = mq;
      }
    }
  }

  /* Return the new trial value. */
  *t = newt;
  return 0;
}

static double owlqn_x1norm(const double* x, const int start, const int end) {
  int i;
  double norm = 0.0;
  for (i = start; i < end; i++) {
    norm += fabs(x[i]);
  }
  return norm;
}

static void owlqn_pseudo_gradient(double* pg, const double* x, const double* g,
                                  const int n, const double c, const int start,
                                  const int end) {
  int i;

  /* Compute the negative of gradients. */
  for (i = 0; i < start; i++) {
    pg[i] = g[i];
  }

  /* Compute the pseudo-gradient. */
  for (i = start; i < end; i++) {
    double xi = x[i];
    double gi = g[i];
    if (xi < 0.0) {
      /* Differentiable. */
      pg[i] = gi - c;
    } else if (xi > 0.0) {
      /* Differentiable. */
      pg[i] = gi + c;
    } else {
      if (gi < -c) {
        /* Take the right partial derivative. */
        pg[i] = gi + c;
      } else if (gi > c) {
        /* Take the left partial derivative. */
        pg[i] = gi - c;
      } else {
        pg[i] = 0.0;
      }
    }
  }

  for (i = end; i < n; i++) {
    pg[i] = g[i];
  }
}

static void owlqn_project(double* d, const double* sign, const int start,
                          const int end) {
  int i;
  for (i = start; i < end; i++)
    if (d[i] * sign[i] <= 0.0) {
      d[i] = 0.0;
    }
}

static void owlqn_contrain_line_search(double* d, const double* pg,
                                       const int start, const int end) {
  int i;
  for (i = start; i < end; i++)
    if (d[i] * pg[i] >= 0) {
      d[i] = 0.0;
    }
}
