#include <stdio.h>
#include <stdlib.h>
#include <functional>
#include <omp.h>
#include "libdcl.h"

double f(std::vector<double> &x_,void *sd)
{
    assert(x_.size()==2);
    double x,y;
    x = x_[0];
    y = x_[1];
    return (1.0-x)*(1.0-x)+(y-x*x)*(y-x*x);
}

double g(double *x_,void *sd)
{
    double x,y;
    x = x_[0];
    y = x_[1];
    return x*x*sin(y);
}

int main()
{
    std::vector<double> x_init={-50.0,5.0};
    std::vector<double> x_out={100.0,100.0};
    std::vector<int> ns,idx;
    size_t offset;
    double_complex_t mu,lambda,rv;
    double a[2]={0,0};
    double b[2]={2,2};
    mu = 1+I;
    lambda = 1+2*I;
    double z = 1.0;
    rv = WhittakerW(lambda,mu,z);
    printf("%.16lg %.16lg\n",creal(rv),cimag(rv));
    find_minimum_descent(f,x_init,x_out,NULL,1);
    printf("%.16lg %.16lg\n",x_out[0],x_out[1]);
    printf("I=%.16lg\n",GaussIntegrateCuba(g,NULL,2,a,b));
    printf("Ai(5)=%.16lg\n",AiryBi(5.0,0));
    ns = {800,600};
    idx = {1,2};
    offset = idx2offset(ns,idx);
    printf("off = %ld\n",offset);
    printf("%d\n",(int)6.9);
    return 0;
}