#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <vector>
#include <ios>
#include <string>
#include <array>
#include <iostream>
#include <assert.h>
#include <cassert>
#include <algorithm>
#include <stdint.h>
#include <complex.h>
#include <thread>
#include <functional>
#include <Eigen/Eigen>
#include <Eigen/Eigenvalues>
#include "libamos.h"
#include "libeispack.h"
#include "cubature.h"
#include <omp.h>
#include "itertools/itertools.hpp"

#ifdef LONG_DOUBLE
typedef long double double_t;
#else
typedef double double_t;
#endif

#if defined(USE_MKL)
#include <mkl.h>
typedef MKL_Complex16 lapack_complex_t;
#else
typedef struct{
    double real;
    double imag;
}lapack_complex_t;
#endif
using complex_t = std::complex<double>;
const complex_t _i(0,1);//мнимая единица
#ifdef USE_MKL
#else
#ifdef __cplusplus
extern "C"{
#endif
//extern lapack routines
void dgetrf_(int* M, int *N, double* A, int* lda, int* IPIV, int* INFO);
void zgetrf_(int* M, int *N, lapack_complex_t* A, int* lda, int* IPIV, int* INFO);
void dgetri_(int* N, double* A, int* lda, int* IPIV, double* WORK, int* lwork, int* INFO);
void zheev_(char const* jobz, char const* uplo,int const* n,  lapack_complex_t* A, int const* lda, double* W, lapack_complex_t* work, int const* lwork,double* rwork,int* info);
void dsyev_(char const* jobz, char const* uplo,int const* n,double* A, int const* lda,double* W,double* work, int const* lwork,int* info);
void zgesv_(int const* n, int const* nrhs,lapack_complex_t* A, int const* lda, int* ipiv, lapack_complex_t* B, int const* ldb,int* info );
void dgtsv_(int const* n, int const* nrhs, double* DL, double* D, double* DU, double* B, int const* ldb,int* info );
void zheevr_( const char* jobz, const char* range, const char* uplo,
              const int* n, lapack_complex_t* a, const int* lda,
              const double* vl, const double* vu, const int* il,
              const int* iu, const double* abstol, int* m, double* w,
              lapack_complex_t* z, const int* ldz, int* isuppz,
              lapack_complex_t* work, const int* lwork, double* rwork,
              const int* lrwork, int* iwork, const int* liwork,
              int* info );
void zgeev_(
    char* jobvl,      // 'N' или 'V': вычислять левые собственные векторы
    char* jobvr,      // 'N' или 'V': вычислять правые собственные векторы
    int* n,           // порядок матрицы
    lapack_complex_t* a,   // матрица [lda, n] (комплексные числа)
    int* lda,         // ведущая размерность матрицы
    lapack_complex_t* w,          // массив собственных значений [n]
    lapack_complex_t* vl,         // левые векторы [ldvl, n]
    int* ldvl,        // ведущая размерность vl
    lapack_complex_t* vr,         // правые векторы [ldvr, n]
    int* ldvr,        // ведущая размерность vr
    void* work,       // рабочий массив
    int* lwork,       // размер work
    double* rwork,    // вещественный рабочий массив
    int* info         // код ошибки
    );
#ifdef __cplusplus
}
#endif
#endif

const double pi=3.14159265358979323846264338327;
const double e_const=     2.71828182845904523536028747135;
const double EulerGamma= 0.577215664901532860606512090082;
const double e0= 4.803242e-10;
const double m0= 9.10938188e-28;
const double hbar= 1.05457159642e-27;
const double kB= 1.3806503e-16;
const double evolt= 1.602176462e-12;
const double angstrem= 1e-8;
const double c_light= 2.99792458e10;
const double Ry=13.605693122990*evolt;
const double a_bohr=5.2917721090380e-9;
const double j_conversion = 2997924536.84314;//Conversion between CGSE and A
const double au = Ry*a_bohr;
const double NaN= 0.0/0.0;
const double Inf= 1.0/0.0;
typedef double __complex__ double_complex_t;
typedef long double __complex__ long_double_complex_t;
typedef std::vector<std::vector<double>> matrix_t;
typedef std::vector<std::vector<double_complex_t>> cmatrix_t;
std::pair<double_complex_t,double_complex_t> HeunT(double_complex_t q, double_complex_t alpha, double_complex_t gamma, double_complex_t delta, double_complex_t epsilon, double z);
double Zeta(double s);
double PolyLog(double s,double z);
double ClebschGordan(double j1,double m1,double j2,double m2,double j3,double m3);
int KronekerDelta(int l1,int l2);
double min(double a,double b);
double max(double a,double b);
double fmax(double a,double b);
double ChebyshevT(int n,double x);
double ChebyshevU(int n,double x);
double HermiteH(int n,double x);
double LaguerreL(int n,int m,double x);
long double LaguerreLa(int n,long double a,double x);
double EllipticE(double m);
double EllipticK(double m);
double_complex_t SphericalHarmonicY(int l,int m,double theta,double phi);
double Gamma(double x);
double_complex_t ZGamma(double_complex_t x);
double ExpIntegralEi(double x);
double BinomCoeff(unsigned int n,unsigned int k);
double LegendrePlm(int l,int m,double x);
double GaussIntegrateParallel(std::function<double(double [],void *)> f,void *serviceData,int ndim,double a[],double b[],int m,unsigned int nproc);
long double GaussIntegrateParallelLong(std::function<long double(long double [],void *)> f, void *serviceData, int ndim, long double a[], long double b[], int m, unsigned int nproc);
double_complex_t ZGaussIntegrateParallel(std::function<double_complex_t(double [],void *)> f,void *serviceData,int ndim,double a[],double b[],int m,unsigned int nproc);
double GaussIntegrateElem(std::function<double(double [],void *)> f,void * serviceData,int ndim,double a[],double b[]);
double_complex_t ZGaussIntegrateElem(std::function<double_complex_t(double [],void *)> f,void * serviceData,int ndim,double a[],double b[]);
double GaussIntegrate(std::function<double(double [],void *)> f,void * serviceData,int ndim,double a[],double b[],int m);
double_complex_t ZGaussIntegrate(std::function<double_complex_t(double [],void *)> f,void * serviceData,int ndim,double a[],double b[],int m);
double GaussIntegrateParallelEpsilon(std::function<double(double [],void *)> f, void *serviceData, int ndim, double a[], double b[], double epsrel, unsigned int nproc);
double_complex_t ZGaussIntegrateParallelEpsilon(std::function<double_complex_t(double [],void *)> f, void *serviceData, int ndim, double a[], double b[], double epsrel, unsigned int nproc);
double GaussIntegrateCuba(std::function<double(double [],void *)> f, void *serviceData, int ndim, double a[], double b[],double epsabs = 1e-32, double epsrel=1e-12,int maxeval=10000);
double_complex_t ZGaussIntegrateCuba(std::function<double_complex_t(double [],void *)> f, void *serviceData, int ndim, double a[], double b[],double epsabs = 1e-32, double epsrel=1e-12,int maxeval=10000);
int IsNaN(double x);
int IsInf(double x);
int sign(double x);
double FindZero(double (*f)(double,void*),double a,double b,void *serviceData,int *errcode);
double FindNZero(double(*f)(double,void *),double x0,double x1,void *serviceData,double sep,int i,int * nf);
long double FindZeroLong(long double (*f)(long double,void*),long double a,long double b,void *serviceData,int *errcode);
long double FindNZeroLong(long double(*f)(long double,void *),long double x0,long double x1,void *serviceData,long double sep,int i,int * nf);

double EulerBeta(double x,double y);
double LegendreP(int n,double x);
double DLegendreP(int n,double x);
void MatrixMatrixMultiply(double *r,double *a,double *b,int m,int n,int k);
double SQR(double x);
double pythag_m(double a,double b );
void zheevr(char jobz, char range, char uplo, int n, lapack_complex_t* a, int lda, double vl, double vu, int il, int iu, double abstol, double* w, lapack_complex_t* z, int ldz, int* info);
void QR_decompose_rotation(double *a,double *q,double *r,int n);
void QR_decompose_reflection(double *a,double *q,double *r,int n);
void QR_solve(int N,double *q,double *r,double *b,double *x);
void printMatrix(double *a,char * title,int m,int n);
void printMatrixZ(double_complex_t *a,char * title,int m,int n);
void inverse(double* A, int N);
int rk4(void (*F)(int,double,double [],double[],void *),int neq,double y[],double t0,double dt,int nsteps,void *serviceData);
int rk4_step(void (*F)(int,double,double [],double[],void *),int neq,double y[],double t,double dt,void *serviceData);
int rk5(void (*F)(int,double,double [],double[],void *),int neq,double y[],double t0,double dt,int nsteps,void *serviceData);
int rk5_step(void (*F)(int,double,double [],double[],void *),int neq,double y[],double t,double dt,void *serviceData);
int zrk4(void (*F)(int,double,double_complex_t [],double_complex_t[],void *),int neq,double_complex_t y[],double t0,double dt,int nsteps,void *serviceData);
int zrk4_step(void (*F)(int,double,double_complex_t [],double_complex_t[],void *),int neq,double_complex_t y[],double t,double dt,void *serviceData);
int zrk5(void (*F)(int,double,double_complex_t [],double_complex_t[],void *),int neq,double_complex_t y[],double t0,double dt,int nsteps,void *serviceData);
int zrk5_step(void (*F)(int, double, double_complex_t[], double_complex_t[], void *), int neq, double_complex_t y[], double t, double dt, void *serviceData);
int fel78(void (*F)(int,double,double [],double[],void *),int neq,double y[],double t0,double t,void *serviceData);
int zfel78(void (*F)(int,double,double_complex_t [],double_complex_t[],void *),int neq,double_complex_t y[],double t0,double t,void *serviceData);

double ipow(double x,int k);
double_complex_t ZGammaIncomplete(double s,double_complex_t z);
double_complex_t ZExpIntegralE(double s,double_complex_t z);
void print_complex(char * msg,double_complex_t z);
double_complex_t besselj(double nu, double_complex_t z);
double_complex_t bessely(double nu, double_complex_t z);
double_complex_t besseli(double nu, double_complex_t z);
double_complex_t besselk(double nu, double_complex_t z);
double diff1d(double (*f)(double, void *), double x0, int orders, double step,void *sd);
double_complex_t zdiff1d(double_complex_t (*f)(double x,void *), double x0, int order,double step,void *sd);
double dbesselj(double nu, double z);
double dbessely(double nu, double z);
double dbesseli(double nu, double z);
double dbesselk(double nu, double z);
double sj(int l,double x);
double sy(int l,double x);
double si(int l,double x);
double sk(int l,double x);
double AiryAi(double x,int derivative);
double AiryBi(double x,int derivative);
double Pochgammer(int n,double alpha);
double_complex_t ZPochgammer(int n,double_complex_t alpha);
double Hypergeometric1F1(double a,double c,double z);
double_complex_t ZHypergeometric1F1(double_complex_t a,double_complex_t c,double z);
long_double_complex_t hypergeometric_1F1(long_double_complex_t a,long_double_complex_t b,long double z, long double eps = 1e-12, int max_iter = 10000);
double AppellF4(double a,double b,double c1,double c2,double x,double y);
double WhittakerM(double lambda,double mu,double z);
double_complex_t WhittakerW(double_complex_t lambda,double_complex_t mu,double z);
double ifact(int k);
int EigenSystemSym(int n,std::vector<std::vector<double>>,std::vector<double_complex_t> &w,std::vector<std::vector<double_complex_t>> &zc,int eigenvectors,int nproc=1);
int EigenSystemHerm(int n,std::vector<std::vector<double_complex_t>>,std::vector<double_complex_t> &w,std::vector<std::vector<double_complex_t>> &zc,int eigenvectors,int nproc=1);
int EigenSystemHerm2(int n,std::vector<std::vector<double_complex_t>>,std::vector<double_complex_t> &w,std::vector<std::vector<double_complex_t>> &zc,int eigenvectors,int nproc=1);
bool is_sym(matrix_t m);
bool is_herm(cmatrix_t m);

int EigenSystemOrig(int n,double *a,double_complex_t *w,double_complex_t *zc,int eigen_vectors);
void print_complex_vector(char * msg,int n,double_complex_t *z);
void print_complex_matrix(char * msg,int n,double_complex_t *z);

bool find_root(std::function<std::vector<double>(std::vector<double> &x,void *user_data)> f, const std::vector<double> &x_init, std::vector<double> &x_out, void *user_data, double epsilon=1e-10, int max_iter=10000);
bool find_minimum_descent(std::function<double(std::vector<double> &x,void *user_data)> f, const std::vector<double> &x_init, std::vector<double> &x_out, void *user_data, double lambda_init=0.01, double epsilon=1e-10, int max_iter=10000);
bool find_minimum_lbfgs(std::function<double(std::vector<double> &x,void *user_data)> f, const std::vector<double> &x_init, std::vector<double> &x_out, void *user_data, double lambda_init=0.01, double epsilon=1e-10, int max_iter=10000, int m=10);
//bool find_minimum_lbfgs(std::function<double(std::vector<double>&, void*)> f, const std::vector<double>& x_init, std::vector<double>& x_out, void* user_data, double epsilon = 1e-12, int max_iter =10000, int m = 5);
bool find_minimum_1d(double(*f) (double &, void *), const double &x_init, double &x_out, void *user_data, double epsilon, int max_iter);
bool linear_regression_basis(std::function<std::vector<double>(std::vector<double>&,void *)>f_basis,std::vector<double> &y,std::vector<std::vector<double>>&x_n,std::vector<double>&w_out,void *user_data);
unsigned int get_num_threads();
double det(int N, double *a);
double_complex_t zdet(int N, double_complex_t *m);
void EigenSystemComplex(std::vector<std::vector<complex_t>> &a,std::vector<complex_t> &w,std::vector<std::vector<complex_t>> &zc,bool eigenvectors);
double Hypergeometric2F1(double a,double b,double c,double z);
void dump_v(std::string &filename,std::vector<double> v);