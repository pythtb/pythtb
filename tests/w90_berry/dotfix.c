/* Fix for the Apple Accelerate <-> gfortran ABI mismatch on complex BLAS
 * dot-product functions. Accelerate implements zdotc/zdotu/cdotc/cdotu in the
 * f2c convention (result returned via a hidden first pointer argument), but
 * gfortran (without -ff2c) expects the complex result returned by value in
 * registers. The mismatch crashes in ZDOTC. We provide register-return
 * implementations and interpose them ahead of Accelerate via
 * DYLD_INSERT_LIBRARIES + DYLD_FORCE_FLAT_NAMESPACE.
 */
#include <complex.h>

static inline int off(int n, int inc){ return inc > 0 ? 0 : (1 - n) * inc; }

double _Complex zdotc_(const int *n, const double _Complex *x, const int *incx,
                       const double _Complex *y, const int *incy){
    int N=*n, ix=*incx, iy=*incy, i, xi=off(N,ix), yi=off(N,iy);
    double _Complex s = 0.0;
    for(i=0;i<N;i++){ s += conj(x[xi]) * y[yi]; xi+=ix; yi+=iy; }
    return s;
}
double _Complex zdotu_(const int *n, const double _Complex *x, const int *incx,
                       const double _Complex *y, const int *incy){
    int N=*n, ix=*incx, iy=*incy, i, xi=off(N,ix), yi=off(N,iy);
    double _Complex s = 0.0;
    for(i=0;i<N;i++){ s += x[xi] * y[yi]; xi+=ix; yi+=iy; }
    return s;
}
float _Complex cdotc_(const int *n, const float _Complex *x, const int *incx,
                      const float _Complex *y, const int *incy){
    int N=*n, ix=*incx, iy=*incy, i, xi=off(N,ix), yi=off(N,iy);
    float _Complex s = 0.0f;
    for(i=0;i<N;i++){ s += conjf(x[xi]) * y[yi]; xi+=ix; yi+=iy; }
    return s;
}
float _Complex cdotu_(const int *n, const float _Complex *x, const int *incx,
                      const float _Complex *y, const int *incy){
    int N=*n, ix=*incx, iy=*incy, i, xi=off(N,ix), yi=off(N,iy);
    float _Complex s = 0.0f;
    for(i=0;i<N;i++){ s += x[xi] * y[yi]; xi+=ix; yi+=iy; }
    return s;
}
