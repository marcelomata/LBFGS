/**
 * Unconstrained Limited memory BFGS(L-BFGS).
 * <p>
 * Forked from https://github.com/chokkan/liblbfgs
 * <p>
 * The MIT License
 * <p>
 * Copyright (c) 1990 Jorge Nocedal
 * Copyright (c) 2007-2010 Naoaki Okazaki
 * Copyright (c) 2014-2017 Yafei Zhang
 * <p>
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 * <p>
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * <p>
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
package lbfgs4j;

import java.util.logging.Logger;

@SuppressWarnings({"SuspiciousNameCombination", "unused", "WeakerAccess"})
public class LBFGS {
    private static Logger log = Logger.getLogger("LBFGS");

    public static final int LBFGS_SUCCESS = 0;
    public static final int LBFGS_CONVERGENCE = 0;
    public static final int LBFGS_CONVERGENCE_DELTA = 1;
    public static final int LBFGS_ALREADY_MINIMIZED = 2;

    public static final int LBFGSERR_LOGICERROR = -1024;
    public static final int LBFGSERR_CANCELED = -1025;
    public static final int LBFGSERR_INVALID_N = -1026;
    public static final int LBFGSERR_INVALID_EPSILON = -1027;
    public static final int LBFGSERR_INVALID_TESTPERIOD = -1028;
    public static final int LBFGSERR_INVALID_DELTA = -1029;
    public static final int LBFGSERR_INVALID_LINESEARCH = -1030;
    public static final int LBFGSERR_INVALID_MINSTEP = -1031;
    public static final int LBFGSERR_INVALID_MAXSTEP = -1032;
    public static final int LBFGSERR_INVALID_FTOL = -1033;
    public static final int LBFGSERR_INVALID_WOLFE = -1034;
    public static final int LBFGSERR_INVALID_GTOL = -1035;
    public static final int LBFGSERR_INVALID_XTOL = -1036;
    public static final int LBFGSERR_INVALID_MAXLINESEARCH = -1037;
    public static final int LBFGSERR_INVALID_ORTHANTWISE = -1038;
    public static final int LBFGSERR_INVALID_ORTHANTWISE_START = -1039;
    public static final int LBFGSERR_INVALID_ORTHANTWISE_END = -1040;
    public static final int LBFGSERR_OUTOFINTERVAL = -1041;
    public static final int LBFGSERR_INCORRECT_TMINMAX = -1042;
    public static final int LBFGSERR_ROUNDING_ERROR = -1043;
    public static final int LBFGSERR_MINIMUMSTEP = -1044;
    public static final int LBFGSERR_MAXIMUMSTEP = -1045;
    public static final int LBFGSERR_MAXIMUMLINESEARCH = -1046;
    public static final int LBFGSERR_MAXIMUMITERATION = -1047;
    public static final int LBFGSERR_WIDTHTOOSMALL = -1048;
    public static final int LBFGSERR_INVALIDPARAMETERS = -1049;
    public static final int LBFGSERR_INCREASEGRADIENT = -1050;
    public static final int LBFGSERR_LINE_SEARCH_FAILED = -1051;

    public static final int LBFGS_LINESEARCH_DEFAULT = 0;
    public static final int LBFGS_LINESEARCH_MORETHUENTE = 0;
    public static final int LBFGS_LINESEARCH_BACKTRACKING_ARMIJO = 1;
    public static final int LBFGS_LINESEARCH_BACKTRACKING_WOLFE = 2;
    public static final int LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE = 3;

    @SuppressWarnings("WeakerAccess")
    public static class Param {
        /**
         * The number of corrections to approximate the inverse hessian matrix.
         * The L-BFGS stores the computation results of previous m iterations
         * to approximate the inverse hessian matrix of the current iteration.
         * It controls the size of the limited memories(corrections).
         * <p>
         * Values less than 3 are not recommended.
         * Large values will result in excessive computing time and memory usage.
         */
        public int m = 6;
        /**
         * Epsilon for convergence test.
         * It determines the accuracy with which the solution is to be found.
         * lbfgs() stops when:
         * ||g(x)|| < epsilon * max(1, ||x||)
         */
        public double epsilon = 1e-5;
        /**
         * Distance for delta-based convergence test.
         * It determines the distance, in iterations, to compute the rate of decrease
         * of f(x).
         * If the it is zero, lbfgs() does not perform the delta-based convergence
         * test.
         */
        public int past = 0;
        /**
         * Delta for convergence test.
         * It determines the minimum rate of decrease of f(x).
         * lbfgs() stops when:
         * (f(x_{k+1}) - f(x_{k})) / f(x_{k}) < delta
         */
        public double delta = 1e-5;
        /**
         * The maximum number of iterations.
         * lbfgs() stops with LBFGSERR_MAXIMUMITERATION
         * when the iteration counter exceeds max_iterations.
         * Zero means a never ending optimization process until convergence or errors.
         */
        public int max_iterations = 0;
        /**
         * The line search algorithm.
         */
        public int linesearch = LBFGS_LINESEARCH_DEFAULT;
        /**
         * The maximum number of trials for the line search per iteration.
         */
        public int max_linesearch = 40;
        /**
         * The minimum step of the line search.
         * <p>
         * This value need not be modified unless
         * the exponents are too large for the machine being used, or unless the
         * problem is extremely badly scaled (in which case the exponents should
         * be increased).
         */
        public double min_step = 1e-20;
        /**
         * The maximum step of the line search.
         * <p>
         * This value need not be modified unless
         * the exponents are too large for the machine being used, or unless the
         * problem is extremely badly scaled (in which case the exponents should
         * be increased).
         */
        public double max_step = 1e20;
        /**
         * A parameter to control the accuracy of the line search.
         * <p>
         * It should be greater than zero and smaller than 0.5.
         */
        public double ftol = 1e-4;
        /**
         * A coefficient for the (strong) Wolfe condition.
         * It is valid only when the backtracking line-search
         * algorithm is used with the Wolfe condition
         * (LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE or
         * LBFGS_LINESEARCH_BACKTRACKING_WOLFE).
         * <p>
         * It should be greater than the ftol parameter and smaller than 1.0.
         */
        public double wolfe = 0.9;
        /**
         * A parameter to control the accuracy of the line search.
         * <p>
         * It should be greater than the ftol parameter (1e-4) and smaller than 1.0.
         * <p>
         * If evaluations of f(x) and g(x) are expensive
         * with respect to the cost of the iteration (when n is very large),
         * it may be advantageous to set it to a small value.
         * <p>
         * A typical small value is 0.1.
         */
        public double gtol = 0.9;
        /**
         * The machine precision for floating-point values.
         * It must be a positive value set by a client program to
         * estimate the machine precision. The line search will terminate
         * with LBFGSERR_ROUNDING_ERROR if the relative width
         * of the interval of uncertainty is less than it.
         */
        public double xtol = 1e-16;
        /**
         * Coefficient for the L1 norm regularization of x.
         * It should be set to zero for standard optimization problems.
         * Setting it to a positive value activates OWLQN method,
         * which minimizes f(x) + C|x|.
         * It is the coefficient C.
         */
        public double orthantwise_c = 0;
        /**
         * Start/end index for computing |x|.
         * They are valid only for OWLQN method (orthantwise_c != 0.0).
         * They specify the first and last indices between which lbfgs() computes |x|.
         */
        public int orthantwise_start = 0;
        public int orthantwise_end = -1;
    }

    public interface IObjectiveFunction {
        /**
         * Callback to provide f(x) and g(x).
         * Obviously, a client program MUST implement it.
         *
         * @param x    Current x.
         * @param g    [OUTPUT] Current g(x).
         * @param step The current step of the line search.
         * @return double       f(x).
         */
        double evaluate(double x[], double g[], double step);
    }

    public interface IProgress {
        /**
         * Callback to receive the optimization progress.
         *
         * @param x          Current x.
         * @param g          Current g(x).
         * @param fx         Current f(x).
         * @param xnorm      The Euclidean norm of the x.
         * @param gnorm      The Euclidean norm of the g.
         * @param step       The line-search step used for this iteration.
         * @param k          The iteration count.
         * @param n_evaluate The number of evaluations of f(x) and g(x).
         * @return int          Zero to continue the optimization process.
         * Non-zero value will cancel the optimization process.
         * Default progress callback never always returns zero.
         */
        int progress(double x[], double g[], double fx,
                     double xnorm, double gnorm,
                     double step, int k, int n_evaluate);
    }

    public static class DefaultProgress implements IProgress {
        @Override
        public int progress(double x[], double g[], double fx,
                            double xnorm, double gnorm,
                            double step, int k, int n_evaluate) {
            log.info(String.format("iteration=%d, fx=%f, xnorm=%f, gnorm=%f, gnorm/xnorm=%g, " +
                            "step=%f, evaluation=%d",
                    k, fx, xnorm, gnorm, gnorm / xnorm, step, n_evaluate));
            return 0;
        }
    }

    public int run(double x[], IObjectiveFunction obj, IProgress prog, Param param) {
        int n = x.length;
        int i, j, k, ls, end, bound, n_evaluate = 0;
        boolean enable_owlqn;
        double step[] = new double[1];
        int m;
        double xp[];
        double g[], gp[], pg[] = null;
        double d[], w[], pf[] = null;
        IterationData lm[], it;
        double ys, yy;
        double xnorm, gnorm, rate, beta;
        double fx[] = new double[1];
        ILineSearch linesearch;

        if (n <= 0) {
            return LBFGSERR_INVALID_N;
        }

        if (prog == null) {
            prog = new DefaultProgress();
        }

        if (param == null) {
            param = new Param();
        }
        m = param.m;
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

        enable_owlqn = (param.orthantwise_c != 0.0);
        if (enable_owlqn) {
            switch (param.linesearch) {
                case LBFGS_LINESEARCH_BACKTRACKING_WOLFE:
                    linesearch = new LineSearchBacktrackingOWLQN();
                    break;
                default:
                    return LBFGSERR_INVALID_LINESEARCH;
            }
        } else {
            switch (param.linesearch) {
                case LBFGS_LINESEARCH_MORETHUENTE:
                    linesearch = new LineSearchMorethuente();
                    break;
                case LBFGS_LINESEARCH_BACKTRACKING_ARMIJO:
                case LBFGS_LINESEARCH_BACKTRACKING_WOLFE:
                case LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE:
                    linesearch = new LineSearchBacktracking();
                    break;
                default:
                    return LBFGSERR_INVALID_LINESEARCH;
            }
        }


        xp = new double[n];
        g = new double[n];
        gp = new double[n];
        d = new double[n];
        w = new double[n];
        if (enable_owlqn) {
            pg = new double[n];
        }

        lm = new IterationData[m];
        for (i = 0; i < m; i++) {
            it = new IterationData();
            it.alpha = 0.0;
            it.s = new double[n];
            it.y = new double[n];
            it.ys = 0.0;
            lm[i] = it;
        }

        if (param.past > 0) {
            pf = new double[param.past];
        }

        log.info("Free memory: " + Runtime.getRuntime().freeMemory() / 1024 / 1024 + "M");
        log.info("Total memory: " + Runtime.getRuntime().totalMemory() / 1024 / 1024 + "M");
        log.info("Max memory: " + Runtime.getRuntime().maxMemory() / 1024 / 1024 + "M");

        fx[0] = obj.evaluate(x, g, 0);
        n_evaluate++;

        if (enable_owlqn) {
            xnorm = owlqn_x1norm(x, param.orthantwise_start, param.orthantwise_end);
            fx[0] += xnorm * param.orthantwise_c;
            owlqn_pseudo_gradient(pg, x, g, n, param.orthantwise_c, param.orthantwise_start, param.orthantwise_end);
        }

        if (pf != null) {
            pf[0] = fx[0];
        }

        if (!enable_owlqn) {
            ncopy(d, g);
        } else {
            ncopy(d, pg);
        }

        xnorm = norm2(x);
        if (!enable_owlqn) {
            gnorm = norm2(g);
        } else {
            gnorm = norm2(pg);
        }
        if (xnorm < 1.0) {
            xnorm = 1.0;
        }
        if (gnorm / xnorm <= param.epsilon) {
            return LBFGS_ALREADY_MINIMIZED;
        }

        step[0] = norm2inv(d);

        k = 1;
        end = 0;

        for (; ; ) {
            copy(xp, x);
            copy(gp, g);
            if (!enable_owlqn) {
                ls = linesearch.search(x, fx, g, d, step, xp, gp, w, obj, param);
            } else {
                ls = linesearch.search(x, fx, g, d, step, xp, pg, w, obj, param);
                owlqn_pseudo_gradient(pg, x, g, n, param.orthantwise_c,
                        param.orthantwise_start, param.orthantwise_end);
            }

            if (ls < 0) {
                copy(x, xp);
                copy(g, gp);
                return ls;
            }

            n_evaluate += ls;

            xnorm = norm2(x);
            if (!enable_owlqn) {
                gnorm = norm2(g);
            } else {
                gnorm = norm2(pg);
            }

            if (prog.progress(x, g, fx[0], xnorm, gnorm, step[0], k, n_evaluate) != 0) {
                return LBFGSERR_CANCELED;
            }

            if (xnorm < 1.0) {
                xnorm = 1.0;
            }
            if (gnorm / xnorm <= param.epsilon) {
                return LBFGS_CONVERGENCE;
            }

            if (pf != null) {
                if (param.past <= k) {
                    rate = (pf[k % param.past] - fx[0]) / fx[0];
                    if (rate < param.delta) {
                        return LBFGS_CONVERGENCE_DELTA;
                    }
                }
                pf[k % param.past] = fx[0];
            }

            if (param.max_iterations != 0 && param.max_iterations < k + 1) {
                return LBFGSERR_MAXIMUMITERATION;
            }

            it = lm[end];
            diff(it.s, x, xp);
            diff(it.y, g, gp);

            ys = dot(it.y, it.s);
            yy = dot(it.y, it.y);
            it.ys = ys;

            bound = (m <= k) ? m : k;
            k++;
            end = (end + 1) % m;

            if (!enable_owlqn) {
                ncopy(d, g);
            } else {
                ncopy(d, pg);
            }

            j = end;
            for (i = 0; i < bound; i++) {
                j = (j + m - 1) % m;
                it = lm[j];
                it.alpha = dot(it.s, d);
                it.alpha /= it.ys;
                add(d, it.y, -it.alpha);
            }

            scale(d, ys / yy);

            for (i = 0; i < bound; i++) {
                it = lm[j];
                beta = dot(it.y, d);
                beta /= it.ys;
                add(d, it.s, it.alpha - beta);
                j = (j + 1) % m;
            }

            if (enable_owlqn) {
                owlqn_constrain_line_search(d, pg, param.orthantwise_start, param.orthantwise_end);
            }

            step[0] = 1.0;
        }
    }

    private interface ILineSearch {
        int search(double x[], double f[], double g[],
                   double s[], double step[], double xp[],
                   double gp[], double wa[],
                   IObjectiveFunction obj,
                   Param param);
    }

    private static class IterationData {
        double alpha;
        double s[];
        double y[];
        double ys;
    }

    private static double min2(double a, double b) {
        return a <= b ? a : b;
    }

    private static double max2(double a, double b) {
        return a >= b ? a : b;
    }

    private static double max3(double a, double b, double c) {
        double ab = a >= b ? a : b;
        return ab >= c ? ab : c;
    }

    private static void copy(double y[], double[] x) {
        for (int i = 0; i < x.length; ++i) {
            y[i] = x[i];
        }
    }

    private static void ncopy(double y[], double[] x) {
        for (int i = 0; i < x.length; ++i) {
            y[i] = -x[i];
        }
    }

    private static void add(double y[], double x[], double c) {
        for (int i = 0; i < x.length; ++i) {
            y[i] = y[i] + c * x[i];
        }
    }

    private static void diff(double[] z, double[] x, double[] y) {
        for (int i = 0; i < x.length; ++i) {
            z[i] = x[i] - y[i];
        }
    }

    private static void scale(double y[], double c) {
        for (int i = 0; i < y.length; ++i) {
            y[i] = c * y[i];
        }
    }

    private static double dot(double x[], double y[]) {
        double dot = 0;
        for (int i = 0; i < x.length; ++i) {
            dot += x[i] * y[i];
        }
        return dot;
    }

    private static double norm2(double x[]) {
        double dot = 0;
        for (int i = 0; i < x.length; ++i) {
            dot += x[i] * x[i];
        }
        return Math.sqrt(dot);
    }

    private static double norm2inv(double x[]) {
        return 1.0 / norm2(x);
    }

    private static double owlqn_x1norm(double x[], int start, int end) {
        int i;
        double norm = 0.0;
        for (i = start; i < end; i++) {
            norm += Math.abs(x[i]);
        }
        return norm;
    }

    private static void owlqn_pseudo_gradient(double pg[], double x[], double g[],
                                              int n, double c, int start, int end) {
        int i;

        for (i = 0; i < start; i++) {
            pg[i] = g[i];
        }

        for (i = start; i < end; i++) {
            double xi = x[i];
            double gi = g[i];
            if (xi < 0.0) {
                pg[i] = gi - c;
            } else if (xi > 0.0) {
                pg[i] = gi + c;
            } else {
                if (gi < -c) {
                    pg[i] = gi + c;
                } else if (gi > c) {
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

    private static void owlqn_project(double d[], double sign[], int start, int end) {
        int i;
        for (i = start; i < end; i++) {
            if (d[i] * sign[i] <= 0.0) {
                d[i] = 0.0;
            }
        }
    }

    private static void owlqn_constrain_line_search(double d[], double pg[],
                                                    int start, int end) {
        int i;
        for (i = start; i < end; i++) {
            if (d[i] * pg[i] >= 0) {
                d[i] = 0.0;
            }
        }
    }

    class LineSearchMorethuente implements ILineSearch {
        @Override
        public int search(double x[], double f[], double g[],
                          double s[], double step[], double xp[],
                          double gp[], double wa[],
                          IObjectiveFunction obj,
                          Param param) {
            int count = 0;
            boolean brackt[] = new boolean[1], stage1;
            int uinfo = 0;
            double dg[] = new double[1];
            double stx[] = new double[1], fx[] = new double[1], dgx[] = new double[1];
            double sty[] = new double[1], fy[] = new double[1], dgy[] = new double[1];
            double fxm[] = new double[1], dgxm[] = new double[1], fym[] = new double[1], dgym[] = new double[1];
            double fm[] = new double[1], dgm[] = new double[1];
            double finit, ftest1, dginit, dgtest;
            double width, prev_width;
            double stmin, stmax;

            if (step[0] <= 0.0) {
                return LBFGSERR_INVALIDPARAMETERS;
            }

            dginit = dot(g, s);

            if (0 < dginit) {
                return LBFGSERR_INCREASEGRADIENT;
            }

            brackt[0] = false;
            stage1 = true;
            finit = f[0];
            dgtest = param.ftol * dginit;
            width = param.max_step - param.min_step;
            prev_width = 2.0 * width;

            stx[0] = sty[0] = 0.0;
            fx[0] = fy[0] = finit;
            dgx[0] = dgy[0] = dginit;

            for (; ; ) {
                if (brackt[0]) {
                    stmin = min2(stx[0], sty[0]);
                    stmax = max2(stx[0], sty[0]);
                } else {
                    stmin = stx[0];
                    stmax = step[0] + 4.0 * (step[0] - stx[0]);
                }

                if (step[0] < param.min_step) {
                    step[0] = param.min_step;
                }
                if (param.max_step < step[0]) {
                    step[0] = param.max_step;
                }

                if ((brackt[0] && ((step[0] <= stmin || stmax <= step[0]) ||
                        param.max_linesearch <= count + 1 || uinfo != 0)) ||
                        (brackt[0] && (stmax - stmin <= param.xtol * stmax))) {
                    step[0] = stx[0];
                }

                copy(x, xp);
                add(x, s, step[0]);

                f[0] = obj.evaluate(x, g, step[0]);
                count++;

                dg[0] = dot(g, s);

                ftest1 = finit + step[0] * dgtest;

                if (brackt[0] && ((step[0] <= stmin || stmax <= step[0]) || uinfo != 0)) {
                    return LBFGSERR_ROUNDING_ERROR;
                }
                if (step[0] == param.max_step && f[0] <= ftest1 && dg[0] <= dgtest) {
                    return LBFGSERR_MAXIMUMSTEP;
                }
                if (step[0] == param.min_step && (ftest1 < f[0] || dgtest <= dg[0])) {
                    return LBFGSERR_MINIMUMSTEP;
                }
                if (brackt[0] && (stmax - stmin) <= param.xtol * stmax) {
                    return LBFGSERR_WIDTHTOOSMALL;
                }
                if (count >= param.max_linesearch) {
                    return LBFGSERR_MAXIMUMLINESEARCH;
                }
                if (f[0] <= ftest1 && Math.abs(dg[0]) <= param.gtol * (-dginit)) {
                    return count;
                }

                if (stage1 && f[0] <= ftest1 &&
                        min2(param.ftol, param.gtol) * dginit <= dg[0]) {
                    stage1 = false;
                }

                if (stage1 && ftest1 < f[0] && f[0] <= fx[0]) {
                    fm[0] = f[0] - step[0] * dgtest;
                    fxm[0] = fx[0] - stx[0] * dgtest;
                    fym[0] = fy[0] - sty[0] * dgtest;
                    dgm[0] = dg[0] - dgtest;
                    dgxm[0] = dgx[0] - dgtest;
                    dgym[0] = dgy[0] - dgtest;

                    uinfo = update_trial_interval(stx, fxm, dgxm, sty, fym, dgym, step,
                            fm, dgm, stmin, stmax, brackt);

                    fx[0] = fxm[0] + stx[0] * dgtest;
                    fy[0] = fym[0] + sty[0] * dgtest;
                    dgx[0] = dgxm[0] + dgtest;
                    dgy[0] = dgym[0] + dgtest;
                } else {
                    uinfo = update_trial_interval(stx, fx, dgx, sty, fy, dgy, step, f,
                            dg, stmin, stmax, brackt);
                }

                if (brackt[0]) {
                    if (0.66 * prev_width <= Math.abs(sty[0] - stx[0])) {
                        step[0] = stx[0] + 0.5 * (sty[0] - stx[0]);
                    }
                    prev_width = width;
                    width = Math.abs(sty[0] - stx[0]);
                }
            }
        }

        private boolean fsigndiff(double x, double y) {
            return x * (y / Math.abs(y)) < 0.0;
        }

        private double CUBIC_MINIMIZER(double u, double fu, double du,
                                       double v, double fv, double dv) {
            d = v - u;
            theta = (fu - fv) * 3 / d + du + dv;
            p = Math.abs(theta);
            q = Math.abs(du);
            r = Math.abs(dv);
            s = max3(p, q, r);
            a = theta / s;
            gamma = s * Math.sqrt(a * a - (du / s) * (dv / s));
            if (v < u) gamma = -gamma;
            p = gamma - du + theta;
            q = gamma - du + gamma + dv;
            r = p / q;
            return u + r * d;
        }

        private double CUBIC_MINIMIZER2(double u, double fu, double du,
                                        double v, double fv, double dv,
                                        double xmin, double xmax) {
            d = v - u;
            theta = (fu - fv) * 3 / d + du + dv;
            p = Math.abs(theta);
            q = Math.abs(du);
            r = Math.abs(dv);
            s = max3(p, q, r);
            a = theta / s;
            gamma = s * Math.sqrt(max2(0, a * a - (du / s) * (dv / s)));
            if (u < v) gamma = -gamma;
            p = gamma - dv + theta;
            q = gamma - dv + gamma + du;
            r = p / q;
            if (r < 0. && gamma != 0.0) {
                return v - r * d;
            } else if (a < 0) {
                return xmax;
            } else {
                return xmin;
            }
        }

        private double QUARD_MINIMIZER(double u, double fu, double du,
                                       double v, double fv) {
            a = v - u;
            return u + du / ((fu - fv) / a + du) / 2 * a;
        }

        private double QUARD_MINIMIZER2(double u, double du,
                                        double v, double dv) {
            a = u - v;
            return v + dv / (dv - du) * a;
        }

        private int update_trial_interval(double x[], double fx[], double dx[], double y[],
                                          double fy[], double dy[], double t[], double ft[],
                                          double dt[], double tmin,
                                          double tmax, boolean brackt[]) {
            boolean bound;
            boolean dsign = fsigndiff(dt[0], dx[0]);
            double mc;
            double mq;
            double newt;

            if (brackt[0]) {
                if (t[0] <= min2(x[0], y[0]) || max2(x[0], y[0]) <= t[0]) {
                    return LBFGSERR_OUTOFINTERVAL;
                }
                if (0.0 <= dx[0] * (t[0] - x[0])) {
                    return LBFGSERR_INCREASEGRADIENT;
                }
                if (tmax < tmin) {
                    return LBFGSERR_INCORRECT_TMINMAX;
                }
            }

            if (fx[0] < ft[0]) {
                brackt[0] = true;
                bound = true;
                mc = CUBIC_MINIMIZER(x[0], fx[0], dx[0], t[0], ft[0], dt[0]);
                mq = QUARD_MINIMIZER(x[0], fx[0], dx[0], t[0], ft[0]);
                if (Math.abs(mc - x[0]) < Math.abs(mq - x[0])) {
                    newt = mc;
                } else {
                    newt = mc + 0.5 * (mq - mc);
                }
            } else if (dsign) {
                brackt[0] = true;
                bound = false;
                mc = CUBIC_MINIMIZER(x[0], fx[0], dx[0], t[0], ft[0], dt[0]);
                mq = QUARD_MINIMIZER2(x[0], dx[0], t[0], dt[0]);
                if (Math.abs(mc - t[0]) > Math.abs(mq - t[0])) {
                    newt = mc;
                } else {
                    newt = mq;
                }
            } else if (Math.abs(dt[0]) < Math.abs(dx[0])) {
                bound = true;
                mc = CUBIC_MINIMIZER2(x[0], fx[0], dx[0], t[0], ft[0], dt[0], tmin, tmax);
                mq = QUARD_MINIMIZER2(x[0], dx[0], t[0], dt[0]);
                if (brackt[0]) {
                    if (Math.abs(t[0] - mc) < Math.abs(t[0] - mq)) {
                        newt = mc;
                    } else {
                        newt = mq;
                    }
                } else {
                    if (Math.abs(t[0] - mc) > Math.abs(t[0] - mq)) {
                        newt = mc;
                    } else {
                        newt = mq;
                    }
                }
            } else {
                bound = false;
                if (brackt[0]) {
                    newt = CUBIC_MINIMIZER(t[0], ft[0], dt[0], y[0], fy[0], dy[0]);
                } else if (x[0] < t[0]) {
                    newt = tmax;
                } else {
                    newt = tmin;
                }
            }

            if (fx[0] < ft[0]) {
                y[0] = t[0];
                fy[0] = ft[0];
                dy[0] = dt[0];
            } else {
                if (dsign) {
                    y[0] = x[0];
                    fy[0] = fx[0];
                    dy[0] = dx[0];
                }
                x[0] = t[0];
                fx[0] = ft[0];
                dx[0] = dt[0];
            }

            if (tmax < newt) {
                newt = tmax;
            }
            if (newt < tmin) {
                newt = tmin;
            }

            if (brackt[0] && bound) {
                mq = x[0] + 0.66 * (y[0] - x[0]);
                if (x[0] < y[0]) {
                    if (mq < newt) {
                        newt = mq;
                    }
                } else {
                    if (newt < mq) {
                        newt = mq;
                    }
                }
            }

            t[0] = newt;
            return 0;
        }

        private double a, d, gamma, theta, p, q, r, s;
    }

    class LineSearchBacktracking implements ILineSearch {
        @Override
        public int search(double x[], double f[], double g[],
                          double s[], double step[], double xp[],
                          double gp[], double wa[],
                          IObjectiveFunction obj,
                          Param param) {
            int count = 0;
            double width, dg;
            double finit, dginit, dgtest;
            double dec = 0.5, inc = 2.1;

            if (step[0] <= 0.0) {
                return LBFGSERR_INVALIDPARAMETERS;
            }

            dginit = dot(g, s);

            if (0 < dginit) {
                return LBFGSERR_INCREASEGRADIENT;
            }

            finit = f[0];
            dgtest = param.ftol * dginit;

            for (; ; ) {
                copy(x, xp);
                add(x, s, step[0]);

                f[0] = obj.evaluate(x, g, step[0]);
                count++;

                if (f[0] > finit + step[0] * dgtest) {
                    width = dec;
                } else {
                    if (param.linesearch == LBFGS_LINESEARCH_BACKTRACKING_ARMIJO) {
                        return count;
                    }

                    dg = dot(g, s);
                    if (dg < param.wolfe * dginit) {
                        width = inc;
                    } else {
                        if (param.linesearch == LBFGS_LINESEARCH_BACKTRACKING_WOLFE) {
                            return count;
                        }

                        if (dg > -param.wolfe * dginit) {
                            width = dec;
                        } else {
                            return count;
                        }
                    }
                }

                if (step[0] < param.min_step) {
                    return LBFGSERR_MINIMUMSTEP;
                }
                if (step[0] > param.max_step) {
                    return LBFGSERR_MAXIMUMSTEP;
                }
                if (count >= param.max_linesearch) {
                    return LBFGSERR_MAXIMUMLINESEARCH;
                }

                step[0] *= width;
            }
        }
    }

    class LineSearchBacktrackingOWLQN implements ILineSearch {
        @Override
        public int search(double x[], double f[], double g[],
                          double s[], double step[], double xp[],
                          double gp[], double wp[],
                          IObjectiveFunction obj,
                          Param param) {
            int i, count = 0;
            double width = 0.5, norm;
            double finit = f[0], dgtest;

            if (step[0] <= 0.0) {
                return LBFGSERR_INVALIDPARAMETERS;
            }

            for (i = 0; i < x.length; i++) {
                wp[i] = (xp[i] == 0.0) ? -gp[i] : xp[i];
            }

            for (; ; ) {
                copy(x, xp);
                add(x, s, step[0]);

                owlqn_project(x, wp, param.orthantwise_start, param.orthantwise_end);

                f[0] = obj.evaluate(x, g, step[0]);
                count++;

                norm = owlqn_x1norm(x, param.orthantwise_start, param.orthantwise_end);
                f[0] += norm * param.orthantwise_c;

                dgtest = 0.0;
                for (i = 0; i < x.length; i++) {
                    dgtest += (x[i] - xp[i]) * gp[i];
                }

                if (f[0] <= finit + param.ftol * dgtest) {
                    return count;
                }

                if (step[0] < param.min_step) {
                    return LBFGSERR_MINIMUMSTEP;
                }
                if (step[0] > param.max_step) {
                    return LBFGSERR_MAXIMUMSTEP;
                }
                if (count >= param.max_linesearch) {
                    return LBFGSERR_MAXIMUMLINESEARCH;
                }

                step[0] *= width;
            }
        }
    }

    public static void main(String args[]) {
        double x[] = new double[1];
        IObjectiveFunction obj1 = new IObjectiveFunction() {
            @Override
            public double evaluate(double[] x, double[] g, double step) {
                double fx = Math.sin(x[0]);
                g[0] = Math.cos(x[0]);
                return fx;
            }
        };
        IObjectiveFunction obj2 = new IObjectiveFunction() {
            @Override
            public double evaluate(double[] x, double[] g, double step) {
                double _x = x[0];
                double fx = _x * _x * _x * _x + _x * _x * _x + _x * _x;
                g[0] = 4 * _x * _x * _x + 3 * _x * _x + 2 * _x;
                return fx;
            }
        };
        IObjectiveFunction obj3 = new IObjectiveFunction() {
            @Override
            public double evaluate(double[] x, double[] g, double step) {
                double exp_nx = Math.exp(-x[0]);
                double fx = 1.0 / (1.0 + exp_nx);
                g[0] = fx * (1.0 - fx);
                return fx;
            }
        };

        LBFGS lbfgs = new LBFGS();
        int ret;
        ret = lbfgs.run(x, obj1, null, null);
        System.out.printf("LBFGS returns %d\n", ret);
        ret = lbfgs.run(x, obj2, null, null);
        System.out.printf("LBFGS returns %d\n", ret);
        ret = lbfgs.run(x, obj3, null, null);
        System.out.printf("LBFGS returns %d\n", ret);
    }
}
