#pragma once
#include <math.h>
#include <stdio.h>
#include <vector>
#include <limits.h>
typedef std::vector<double> double_vector_t;
typedef std::vector<int> int_vector_t;
#define PI 3.14159265358979323846
#define IntegerMAX_VALUE 2147483647
#define DoubleMAX_VALUE 1.7976931348623157E+308
/*ZBESH COMPUTES AN N MEMBER SEQUENCE OF COMPLEX C HANKEL (BESSEL) FUNCTIONS CY(J)=H(M,FNU+J-1,Z)*/
void zbesh(double zr, double zi, double fnu, int kode, int m, int n,
           double_vector_t &cyr, double_vector_t &cyi, int_vector_t &nz, int_vector_t &ierr);

/*BESSEL_I*/
void zbesi(double zr, double zi, double fnu, int kode, int n, double_vector_t &cyr,
           double_vector_t &cyi, int_vector_t &nz, int_vector_t &ierr);

/*BESSEL_J*/
void zbesj(double zr, double zi, double fnu, int kode, int n, double_vector_t &cyr,
           double_vector_t &cyi, int_vector_t &nz, int_vector_t &ierr);

/*BESSEL_K*/
void zbesk(double zr, double zi, double fnu, int kode, int n,
           double_vector_t &cyr, double_vector_t &cyi, int_vector_t &nz, int_vector_t &ierr);

/*BESSEL_Y*/
//cwrkr.resize(sequenceNumber);
//cwrki.resize(sequenceNumber);
//вызвать перед zbesy
void zbesy(double zr, double zi, double fnu, int kode, int n,
           double_vector_t &cyr, double_vector_t &cyi, int_vector_t &nz, double_vector_t &cwrkr, double_vector_t &cwrki,
           int_vector_t &ierr);

/*AIRY_AI*/
void zairy(double zr, double zi, int id, int kode, double_vector_t &air,
           double_vector_t &aii, int_vector_t &nz, int_vector_t &ierr);
/*AIRY_BI*/
void zbiry(double zr, double zi, int id, int kode, double_vector_t &bir,
           double_vector_t &bii, int_vector_t &ierr);


/**/

void zuoik(double zr, double zi, double fnu, int kode, int ikflg, int n,
           double_vector_t &yr, double_vector_t &yi, int_vector_t &nuf);
double zabs(double zr, double zi);
void zbknu(double zr, double zi, double fnu, int kode, int n, double_vector_t &yr,
           double_vector_t &yi, int_vector_t &nz);
void zacon(double zr, double zi, double fnu, int kode, int mr, int n,
           double_vector_t &yr, double_vector_t &yi, int_vector_t &nz);
void zbunk(double zr, double zi, double fnu, int kode, int mr, int n,
           double_vector_t &yr, double_vector_t &yi, int_vector_t &nz);
void zunik(double zrr, double zri, double fnu, int ikflg, int ipmtr,
           int init[], double_vector_t &phir, double_vector_t &phii, double_vector_t &zeta1r, double_vector_t &zeta1i,
           double_vector_t &zeta2r, double_vector_t &zeta2i, double_vector_t &sumr, double_vector_t &sumi,
           double_vector_t &cwrkr, double_vector_t &cwrki);
void zunhj(double zr, double zi, double fnu, int ipmtr, double_vector_t &phir,
           double_vector_t &phii, double_vector_t &argr, double_vector_t &argi, double_vector_t &zeta1r,
           double_vector_t &zeta1i, double_vector_t &zeta2r, double_vector_t &zeta2i, double_vector_t &asumr,
           double_vector_t &asumi, double_vector_t &bsumr, double_vector_t &bsumi);
void zlog(double ar, double ai, double_vector_t &br, double_vector_t &bi, int_vector_t &ierr);
void zuchk(double yr, double yi, int_vector_t &nz, double ascle);
void zshch(double zr, double zi, double_vector_t &cshr, double_vector_t &cshi, double_vector_t &cchr,
           double_vector_t &cchi);
double dgamln(double z, int_vector_t &ierr);
void zexp(double ar, double ai, double_vector_t &br, double_vector_t &bi);
void zdiv(double ar, double ai, double br, double bi, double_vector_t &cr,
          double_vector_t &ci);
void zmlt(double ar, double ai, double br, double bi, double_vector_t &cr,
          double_vector_t &ci);
void zsqrt(double ar, double ai, double_vector_t &br, double_vector_t &bi);
void zkscl(double zrr, double zri, double fnu, int n, double_vector_t &yr,
           double_vector_t &yi, int_vector_t &nz, double rzr, double rzi, double ascle);
void zbinu(double zr, double zi, double fnu, int kode, int n,
           double_vector_t &cyr, double_vector_t &cyi, int_vector_t &nz);
void zs1s2(double zrr, double zri, double_vector_t &s1r, double_vector_t &s1i, double_vector_t &s2r,
           double_vector_t &s2i, int_vector_t &nz, double ascle, int_vector_t &iuf);
void zunk1(double zr, double zi, double fnu, int kode, int mr, int n,
           double_vector_t &yr, double_vector_t &yi, int_vector_t &nz);
void zunk2(double zr, double zi, double fnu, int kode, int mr, int n,
           double_vector_t &yr, double_vector_t &yi, int_vector_t &nz);
void zseri(double zr, double zi, double fnu, int kode, int n, double_vector_t &yr,
           double_vector_t &yi, int_vector_t &nz);
void zacai(double zr, double zi, double fnu, int kode, int mr, int n,
           double_vector_t &yr, double_vector_t &yi, int_vector_t &nz);
void zmlri(double zr, double zi, double fnu, int kode, int n, double_vector_t &yr,
           double_vector_t &yi, int_vector_t &nz);
void zasyi(double zr, double zi, double fnu, int kode, int n, double_vector_t &yr,
           double_vector_t &yi, int_vector_t &nz);
void zwrsk(double zrr, double zri, double fnu, int kode, int n,
           double_vector_t &yr, double_vector_t &yi, int_vector_t &nz, double_vector_t &cwr, double_vector_t &cwi);
void zbuni(double zr, double zi, double fnu, int kode, int n, double_vector_t &yr,
           double_vector_t &yi, int_vector_t &nz, int nui, int_vector_t &nlast, double fnul);
void zrati(double zr, double zi, double fnu, int n, double_vector_t &cyr,
           double_vector_t &cyi);
void zuni1(double zr, double zi, double fnu, int kode, int n, double_vector_t &yr,
           double_vector_t &yi, int_vector_t &nz, int_vector_t &nlast, double fnul);
void zuni2(double zr, double zi, double fnu, int kode, int n, double_vector_t &yr,
           double_vector_t &yi, int_vector_t &nz, int_vector_t &nlast, double fnul);

void zbesyh(double zr, double zi, double fnu, int kode, int n,
            double_vector_t &cyr, double_vector_t &cyi, int_vector_t &nz, double_vector_t &cwrkr, double_vector_t &cwrki,
            int_vector_t &ierr);

/**************/
void zqcbh(int mqc);
void zqcbj(int mqc);
void zqcbk(int mqc);
void zqcby(int mqc);
void zqcai(int mqc);
