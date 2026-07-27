#include "libdcl.h"
#include <stdint.h>
#include <cmath>
#include <limits>

#include "libamos.h"
#define N_max 200
#define N_trunc 15
#define nu_max 15

typedef struct{
    double_complex_t (*F)(double,void *);
    double omega;
    void *sd;
}fourier_param;

typedef struct{
    std::function<double(double [],void *)> f;
    void *sd;
    int ndim;
    double *a;
    double *b;
}cuba_t;

typedef struct{
    std::function<double_complex_t(double [],void *)> f;
    void *sd;
    int ndim;
    double *a;
    double *b;
}zcuba_t;

typedef struct{
    double s;
    double z;
}polylog_param;

typedef struct{
    double_complex_t p;
    double k;
}serviceData;

typedef struct{
    double mu;
    double lambda;
    double z;
}w_params;

typedef struct{
    double_complex_t mu;
    double_complex_t lambda;
    double z;
}w_params_complex;

typedef struct{
    double alpha;
    double beta1;
    double beta2;
    double gamma1;
    double gamma2;
    double x;
    double y;
}f2_params;

typedef struct
{
    double a;
    double b;
    double c;
    double z;
}hypergeom2f1_params;

typedef struct
{
    double a;
    double b;
    double c1;
    double c2;
    double z1;
    double z2;
}appellf4_params;

typedef struct{
    std::function<double(double [],void *)> *f;
    double **x_glob;
    double *c_glob;
    void *serviceData;
    int ndim;
    long long int size;
    int nproc;
    int tid;
}pt_params;

typedef struct{
    std::function<long double(long double [],void *)> *f;
    long double **x_glob;
    long double *c_glob;
    void *serviceData;
    int ndim;
    long long int size;
    int nproc;
    int tid;
}pt_params_long;


typedef struct{
    std::function<double_complex_t(double [],void *)> *f;
    double **x_glob;
    double *c_glob;
    void *serviceData;
    int ndim;
    long long int size;
    int nproc;
    int tid;
}pt_params_z;

typedef struct{
    double_complex_t q;
    double_complex_t alpha;
    double_complex_t gamma;
    double_complex_t  delta;
    double_complex_t epsilon;
}heun_params_t;

//This function is used to increment indexes in Gauss integrate//
//TODO fix recursion in orthogonal polynoms.
//TODO add parallelism in integration
int increment(int *array,int N,int nmax){
    int i,carry;
    carry=1;
    for(i=0;(i<N)&&carry;i++){
        if(array[i]==nmax){
            array[i]=0;
            carry=1;
        }else{
            array[i]++;
            carry=0;
        }
    }
    return carry;
}

//F(neq,t,y[],rv[]);
void to_H(int,double z,double_complex_t y[] ,double_complex_t rv[],void * sd)
{
    heun_params_t *p;
    p = (heun_params_t *) sd;
    rv[0] = y[1];
    rv[1] = -(p->gamma + p->delta * z + p->epsilon * z *z) * y[1] - (p->alpha * z - p->q) * y[0];
}

/**
 * @brief HeunT
 * Реализация три-конфлюэнтной функции Гойна.
 * Предполагается получить эту функцию через решение дифф. уравнения.
 * @param q
 * Параметр
 * @param alpha
 * Параметр
 * @param gamma
 * Параметр
 * @param delta
 * Параметр
 * @param epsilon
 * Параметр
 * @param z
 * Параметр
 * @return
 * Выход: пара значений функции и ее производной.
 */
std::pair<double_complex_t,double_complex_t> HeunT(double_complex_t q, double_complex_t alpha, double_complex_t gamma,double_complex_t  delta, double_complex_t epsilon, double z)
{
    int N;
    heun_params_t s;
    double_complex_t y[2];
    std::pair<double_complex_t,double_complex_t> result;
    double dz;
    s.alpha=alpha;
    s.delta=delta;
    s.epsilon=epsilon;
    s.gamma=gamma;
    s.q=q;
    y[0] = 1;
    y[1] = 0;
    zfel78(to_H,2,y,0,z,(void *)&s);
    result.first = y[0];
    result.second = y[1];
    return result;
}
//Use the Lanczcos approximation
//TODO: fix the coefficients
double Gamma(double x){
    static const double d[15]=
        {0.011748482397059865854194497518649980956864516461236210282662113866005188,
         0.67149902860259779045990250413501961046434632677594141601778811910717214,
         -0.70018558813697759056405597501837691149234060570108385068405304408801,
         0.1660776982193675198571572139135855322686605807017872543338482203672,
         -0.00577924080929345179860093218662419128059386859373306997961415072,
         3.993855469407750719798353396430172688531170874259718037933e-7,
         5.46582035496776954711515712918817773446478503007554770768e-7,
         -1.155750791439381406933196280628910251054174145609947511436e-6,
         1.85730234701191102358179289192258099062275696389504662339e-6,
         -2.4702880923232733952229889270160370646909100217280405525e-6,
         2.55458552585000269416928843452846310351308677477529019125e-6,
         -1.93048838216538385557473513898173458259530760512112460056e-6,
         9.9178601846535382019593795774495103059935300410647600095e-7,
         -3.07702603925219628485419424700998229231072885973688462704e-7,
         4.3350939794140517541113396050506568511274887170176440299e-8};
    static const double r=607.0/128.0;
    double rv;
    double sum;
    int i;
    if(x<0.5){
        return (pi/sin(pi*x))/Gamma(1-x);
    }
    //Uncomment if you want better precision. This is slower.
    /*if(x>3.0){
    return (x-1)*Gamma(x-1);
  }*/
    rv=2.0*sqrt(e_const/pi)*pow((x+r-0.5)/e_const,x-0.5);
    sum=d[0];
    for(i=1;i<15;i++){
        sum+=d[i]/(x-1+i);
    }
    rv=rv*sum;
    return rv;
}

double_complex_t ZGamma(double_complex_t x){
    static const double d[15]=
        {0.011748482397059865854194497518649980956864516461236210282662113866005188,
         0.67149902860259779045990250413501961046434632677594141601778811910717214,
         -0.70018558813697759056405597501837691149234060570108385068405304408801,
         0.1660776982193675198571572139135855322686605807017872543338482203672,
         -0.00577924080929345179860093218662419128059386859373306997961415072,
         3.993855469407750719798353396430172688531170874259718037933e-7,
         5.46582035496776954711515712918817773446478503007554770768e-7,
         -1.155750791439381406933196280628910251054174145609947511436e-6,
         1.85730234701191102358179289192258099062275696389504662339e-6,
         -2.4702880923232733952229889270160370646909100217280405525e-6,
         2.55458552585000269416928843452846310351308677477529019125e-6,
         -1.93048838216538385557473513898173458259530760512112460056e-6,
         9.9178601846535382019593795774495103059935300410647600095e-7,
         -3.07702603925219628485419424700998229231072885973688462704e-7,
         4.3350939794140517541113396050506568511274887170176440299e-8};
    static const double r=607.0/128.0;
    double_complex_t rv;
    double_complex_t sum;
    int i;
    if(creal(x)<0.5){
        return (pi/csin(pi*x))/ZGamma(1-x);
    }
    //Uncomment if you want better precision. This is slower.
    /*if(x>3.0){
    return (x-1)*Gamma(x-1);
  }*/
    rv=2.0*sqrt(e_const/pi)*cpow((x+r-0.5)/e_const,x-0.5);
    sum=d[0];
    for(i=1;i<15;i++){
        sum+=d[i]/(x-1+i);
    }
    rv=rv*sum;
    return rv;
}


double LegendrePlm(int l,int m,double x){
    double fact1,fact2,rv,sq;
    int i;
    if((abs(m)>abs(l)&&l>=0)||(abs(m)>abs(-l-1)&&l<0)){
        return 0;
    }
    if(l<0){
        return LegendrePlm(-l-1,m,x);
    }
    if(m<0){
        fact1=1.0;
        fact2=1.0;
        rv=1.0;
        for(i=1;i<=l+m;i++){
            fact1*=i;
        }
        for(i=1;i<=l-m;i++){
            fact2*=i;
        }
        if(m%2){
            rv=-rv;
        }
        return rv*fact1*LegendrePlm(l,-m,x)/fact2;
    }
    if(m==l){
        rv=1.0;
        if(m%2){
            rv=-rv;
        }
        for(i=1;i<=(2*m-1);i+=2){
            rv*=i;
        }
        sq=sqrt(1-x*x);
        for(i=0;i<m;i++){
            rv*=sq;
        }
        return rv;
    }
    return (x*(2.0*l-1)*LegendrePlm(l-1,m,x)-(l+m-1)*LegendrePlm(l-2,m,x))/(l-m);
}

double EllipticE(double m){
    double a[20];
    double b[20];
    double c[20];
    double k,e,epsilon,sum,pow;
    int N=19;
    int i;
    if((m<0)||(m>1)){
        return 0;
    }
    if(m==1.0){
        k=1.0/0.0;
        e=1.0;
    }else{
        a[0]=1.0;
        b[0]=sqrt(1.0-m);
        c[0]=sqrt(m);
        //err=1.0;
        epsilon=1e-16;
        for(i=1;i<=N;i++){
            a[i]=(a[i-1]+b[i-1])*0.5;
            b[i]=sqrt(a[i-1]*b[i-1]);
            c[i]=(a[i-1]-b[i-1])*0.5;
            //err=c[i];
        }
        i--;
        if(fabs(c[i])>epsilon){
            printf("Elliptic does not converge after %d steps. STOP\n",N);
            return 0;
        }
        k=pi/(2*a[i]);
        sum=0;
        pow=1;
        for(i=0;i<=N;i++){
            sum+=c[i]*c[i]*pow;
            pow*=2;
        }
        sum*=0.5;
        e=(1-sum)*k;
    }
    return e;
}

double EllipticK(double m){
    double a[20];
    double b[20];
    double c[20];
    double k,e,epsilon,sum,pow;
    int N=19;
    int i;
    if((m<0)||(m>1)){
        return 0;
    }
    if(m==1.0){
        k=1.0/0.0;
        e=1.0;
    }else{
        a[0]=1.0;
        b[0]=sqrt(1.0-m);
        c[0]=sqrt(m);
        //err=1.0;
        epsilon=1e-16;
        for(i=1;i<=N;i++){
            a[i]=(a[i-1]+b[i-1])*0.5;
            b[i]=sqrt(a[i-1]*b[i-1]);
            c[i]=(a[i-1]-b[i-1])*0.5;
            //err=c[i];
        }
        i--;
        if(fabs(c[i])>epsilon){
            printf("Elliptic does not converge after %d steps. STOP\n",N);
            return 0;
        }
        k=pi/(2*a[i]);
        sum=0;
        pow=1;
        for(i=0;i<=N;i++){
            sum+=c[i]*c[i]*pow;
            pow*=2;
        }
        sum*=0.5;
        e=(1-sum)*k;
    }
    return k;
}

double_complex_t SphericalHarmonicY(int l,int m,double theta,double phi){
    double fact1,fact2;
    int i;
    if(l<0){
        return 0;
    }
    fact1=1;
    fact2=1;
    for(i=1;i<=(l-m);i++){
        fact1*=i;
    }
    for(i=1;i<=(l+m);i++){
        fact2*=i;
    }
    return sqrt((2.0*l+1)*fact1/(4*pi*fact2))*LegendrePlm(l,m,cos(theta))*cexp(I*m*phi);
}

double HermiteH(int n,double x){
    if(n<0){
        return 0;
    }
    if(n==0){
        return 1;
    }
    if(n==1){
        return 2*x;
    }
    return 2*x*HermiteH(n-1,x)-2*(n-1)*HermiteH(n-2,x);
}

/**
 * @brief diff1d
 * Формулы численного дифференцирования.
 * @param order
 * Порядок дифференцирования
 * @return
 */
double diff1d(double (*f)(double x,void *), double x0, int order,double step,void *sd)
{
    switch (order) {
    case 0:
        return f(x0,sd);
        break;
    case 1:
        return (f(x0+step,sd)-f(x0-step,sd))/(2*step);
        break;
    case 2:
        return (f(x0+step,sd)+f(x0-step,sd)-2*f(x0,sd))/(step*step);
        break;
    default:
        return NaN;
        break;
    }
}

/**
 * @brief zdiff1d
 * Формулы численного дифференцирования.
 * @param order
 * Порядок дифференцирования
 * @return
 */
double_complex_t zdiff1d(double_complex_t (*f)(double x,void *), double x0, int order,double step,void *sd)
{
    switch (order) {
    case 0:
        return f(x0,sd);
        break;
    case 1:
        return (f(x0+step,sd)-f(x0-step,sd))/(2*step);
        break;
    case 2:
        return (f(x0+step,sd)+f(x0-step,sd)-2*f(x0,sd))/(step*step);
        break;
    default:
        return NaN;
        break;
    }
}

double ChebyshevT(int n,double x){
    if(n<0){
        return 0;
    }
    if(n==0){
        return 1;
    }
    if(n==1){
        return x;
    }
    return 2*x*ChebyshevT(n-1,x)-ChebyshevT(n-2,x);
}

double ChebyshevU(int n,double x){
    if(n<0){
        return 0;
    }
    if(n==0){
        return 1;
    }
    if(n==1){
        return 2*x;
    }
    return 2*x*ChebyshevU(n-1,x)-ChebyshevU(n-2,x);
}

double BinomCoeff(unsigned int n,unsigned int k){
    int i;
    double rv;
    rv=1.0;
    if(k<=n){
        for(i=1;i<=n;i++){
            rv*=i;
        }
        for(i=1;i<=k;i++){
            rv/=i;
        }
        for(i=1;i<=(n-k);i++){
            rv/=i;
        }
    }else{
        rv=0.0;
    }
    return rv;
}

double ExpIntegralEi(double x){
    double epsilon,xm,rv,sum,fact,term,prev;
    int i,maxiter;
    epsilon=1e-17;
    xm=fabs(log(epsilon));
    maxiter=120;
    if(fabs(x)<xm){
        //Do power series
        rv=EulerGamma+log(fabs(x));
        sum=0;
        fact=1.0;
        for(i=1;i<maxiter;i++){
            fact*=x/i;
            term=fact/i;
            sum+=term;
            if(fabs(term)<epsilon*sum) break;
        }
        rv+=sum;
    }else{
        //Do asymptotic expansion
        sum=0;
        term=1.0;
        for(i=1;i<maxiter;i++){
            prev=term;
            term*=i/x;
            if(fabs(term)<epsilon) break;
            if(fabs(term)<fabs(prev)){
                sum+=term;
            }else{
                sum-=prev;
                break;
            }
        }
        rv=exp(x)*(1.0+sum)/x;
    }
    return rv;
}

//m difines number of sections;
double GaussIntegrate(std::function<double(double [],void *)> f,void *serviceData,int ndim,double a[],double b[],int m){
    int *idx;
    int carry;
    double rv;
    double *a1,*b1;
    double delta;
    int i;
    idx=(int *)malloc(ndim*sizeof(int));
    a1=(double *)malloc(ndim*sizeof(double));
    b1=(double *)malloc(ndim*sizeof(double));
    carry=0;
    rv=0;
    for(i=0;i<ndim;i++){
        idx[i]=0;
    }
    delta=1.0/(double)m;
    while(!carry){
        for(i=0;i<ndim;i++){
            a1[i]=delta*idx[i]*(b[i]-a[i])+a[i];
            b1[i]=(idx[i]+1)*delta*(b[i]-a[i])+a[i];
        }
        rv+=GaussIntegrateElem(f,serviceData,ndim,a1,b1);
        carry=increment(idx,ndim,m-1);
    }
    free(a1);
    free(b1);
    free(idx);
    return rv;
}

//m difines number of sections;
double_complex_t ZGaussIntegrate(std::function<double_complex_t(double [],void *)> f,void *serviceData,int ndim,double a[],double b[],int m){
    int *idx;
    int carry;
    double_complex_t rv;
    double *a1,*b1;
    double delta;
    int i;
    idx=(int *)malloc(ndim*sizeof(int));
    a1=(double *)malloc(ndim*sizeof(double));
    b1=(double *)malloc(ndim*sizeof(double));
    carry=0;
    rv=0;
    for(i=0;i<ndim;i++){
        idx[i]=0;
    }
    delta=1.0/(double)m;
    while(!carry){
        for(i=0;i<ndim;i++){
            a1[i]=delta*idx[i]*(b[i]-a[i])+a[i];
            b1[i]=(idx[i]+1)*delta*(b[i]-a[i])+a[i];
        }
        rv+=ZGaussIntegrateElem(f,serviceData,ndim,a1,b1);
        carry=increment(idx,ndim,m-1);
    }
    free(a1);
    free(b1);
    free(idx);
    return rv;
}

double GaussIntegrateElem(std::function<double(double [],void *)> f,void *serviceData,int ndim,double a[],double b[]){
    //define Gauss rule;
    static const double ksi[15]={-0.9879925180204854,-0.937273392400706,-0.8482065834104272,-0.7244177313601701,-0.5709721726085388,-0.3941513470775634,-0.2011940939974345,0,0.2011940939974345,0.3941513470775634,0.5709721726085388,0.7244177313601701,0.8482065834104272,0.937273392400706,0.9879925180204854};
    static const double an[15]={0.03075324199611749,0.07036604748810815,0.107159220467172,0.1395706779261543,0.1662692058169939,0.1861610000155622,0.1984314853271116,0.2025782419255613,0.1984314853271116,0.1861610000155622,0.1662692058169939,0.1395706779261543,0.107159220467172,0.07036604748810815,0.03075324199611749};
    int nmax=14;
    int *idx;
    double *x;
    double rv,eta;
    int i,j,carry;
    idx=(int *)malloc(ndim*sizeof(int));
    x=(double *)malloc(ndim*sizeof(double));
    for(i=0;i<ndim;i++){
        idx[i]=0;
    }
    rv=0.0;
    carry=0;
    while(!carry){
        for(i=0;i<ndim;i++){
            x[i]=ksi[idx[i]]*(b[i]-a[i])/2.0+(a[i]+b[i])/2.0;
        }
        eta=f(x,serviceData);
        for(i=0;i<ndim;i++){
            eta=eta*an[idx[i]]*(b[i]-a[i])/2.0;
        }
        rv+=eta;
        carry=increment(idx,ndim,nmax);
    }
    free(idx);
    free(x);
    return rv;
}

double_complex_t ZGaussIntegrateElem(std::function<double_complex_t(double [],void *)> f,void *serviceData,int ndim,double a[],double b[]){
    //define Gauss rule;
    static const double ksi[15]={-0.9879925180204854,-0.937273392400706,-0.8482065834104272,-0.7244177313601701,-0.5709721726085388,-0.3941513470775634,-0.2011940939974345,0,0.2011940939974345,0.3941513470775634,0.5709721726085388,0.7244177313601701,0.8482065834104272,0.937273392400706,0.9879925180204854};
    static const double an[15]={0.03075324199611749,0.07036604748810815,0.107159220467172,0.1395706779261543,0.1662692058169939,0.1861610000155622,0.1984314853271116,0.2025782419255613,0.1984314853271116,0.1861610000155622,0.1662692058169939,0.1395706779261543,0.107159220467172,0.07036604748810815,0.03075324199611749};
    int nmax=14;
    int *idx;
    double *x;
    double_complex_t rv,eta;
    int i,j,carry;
    idx=(int *)malloc(ndim*sizeof(int));
    x=(double *)malloc(ndim*sizeof(double));
    for(i=0;i<ndim;i++){
        idx[i]=0;
    }
    rv=0.0;
    carry=0;
    while(!carry){
        for(i=0;i<ndim;i++){
            x[i]=ksi[idx[i]]*(b[i]-a[i])/2.0+(a[i]+b[i])/2.0;
        }
        eta=f(x,serviceData);
        for(i=0;i<ndim;i++){
            eta=eta*an[idx[i]]*(b[i]-a[i])/2.0;
        }
        rv+=eta;
        carry=increment(idx,ndim,nmax);
    }
    free(idx);
    free(x);
    return rv;
}

void * pt_proc(void *params){
    pt_params *p;
    double **x_glob;
    double *c_glob;
    std::function<double(double [],void *)> *f;
    // double (*f)(double *,void *);
    void *serviceData;
    int ndim,tid,nproc;
    long long int size,i;
    double rv=0;
    double *rvptr;
    p=(pt_params *)params;
    x_glob=p->x_glob;
    c_glob=p->c_glob;
    f=p->f;
    ndim=p->ndim;
    size=p->size;
    tid=p->tid;
    nproc=p->nproc;
    serviceData=p->serviceData;
    for(i=tid;i<size;i+=nproc){
        rv+=(*f)(x_glob[i],serviceData)*c_glob[i];
    }
    rvptr=(double *)malloc(sizeof(double));
    *rvptr=rv;
    return (void *)rvptr;
}

void * pt_proc_long(void *params){
    pt_params_long *p;
    long double **x_glob;
    long double *c_glob;
    std::function<long double(long double [],void *)> *f;
    // long double (*f)(long double *,void *);
    void *serviceData;
    int ndim,tid,nproc;
    long long int size,i;
    long double rv=0;
    long double *rvptr;
    p=(pt_params_long *)params;
    x_glob=p->x_glob;
    c_glob=p->c_glob;
    f=p->f;
    ndim=p->ndim;
    size=p->size;
    tid=p->tid;
    nproc=p->nproc;
    serviceData=p->serviceData;
    for(i=tid;i<size;i+=nproc){
        rv+=(*f)(x_glob[i],serviceData)*c_glob[i];
    }
    rvptr=(long double *)malloc(sizeof(long double));
    *rvptr=rv;
    return (void *)rvptr;
}

void * pt_proc_z(void *params){
    pt_params_z *p;
    double **x_glob;
    double *c_glob;
    // double_complex_t (*f)(double *,void *);
    std::function<double_complex_t(double [],void *)> *f;
    void *serviceData;
    int ndim,tid,nproc;
    long long int size,i;
    double_complex_t rv=0;
    double_complex_t *rvptr;
    p=(pt_params_z *)params;
    x_glob=p->x_glob;
    c_glob=p->c_glob;
    f=p->f;
    ndim=p->ndim;
    size=p->size;
    tid=p->tid;
    nproc=p->nproc;
    serviceData=p->serviceData;
    for(i=tid;i<size;i+=nproc){
        rv+=(*f)(x_glob[i],serviceData)*c_glob[i];
    }
    rvptr=(double_complex_t *)malloc(sizeof(double_complex_t));
    *rvptr=rv;
    return (void *)rvptr;
}

unsigned int get_num_threads(){
    return std::thread::hardware_concurrency();
}

//Realize cubature of Gauss Rule using pthreads.
double GaussIntegrateParallel(std::function<double(double [],void *)> f, void *serviceData, int ndim, double a[], double b[], int m, unsigned int nproc){
    pthread_t *tids;
    pt_params *tparams;
    int *idx,*idx_current;
    int carry,carry_current;
    double rv,eta;
    double *a1,*b1;
    double *x;
    double **x_glob,*c_glob;
    double **retval;
    long long int size;
    double delta;
    int i,j;
    //define Gauss rule;
    static const double ksi[15]={-0.9879925180204854,-0.937273392400706,-0.8482065834104272,-0.7244177313601701,-0.5709721726085388,-0.3941513470775634,-0.2011940939974345,0,0.2011940939974345,0.3941513470775634,0.5709721726085388,0.7244177313601701,0.8482065834104272,0.937273392400706,0.9879925180204854};
    static const double an[15]={0.03075324199611749,0.07036604748810815,0.107159220467172,0.1395706779261543,0.1662692058169939,0.1861610000155622,0.1984314853271116,0.2025782419255613,0.1984314853271116,0.1861610000155622,0.1662692058169939,0.1395706779261543,0.107159220467172,0.07036604748810815,0.03075324199611749};
    int nmax=14;
    long long int counter=0;
    idx_current=(int *)malloc(ndim*sizeof(int));
    x=(double *)malloc(ndim*sizeof(double));
    idx=(int *)malloc(ndim*sizeof(int));
    a1=(double *)malloc(ndim*sizeof(double));
    b1=(double *)malloc(ndim*sizeof(double));
    retval=(double **)malloc(nproc*sizeof(double *));
    tids=(pthread_t *)malloc(nproc*sizeof(pthread_t));
    tparams=(pt_params *)malloc(nproc*sizeof(pt_params));
    carry=0;
    carry_current=0;
    rv=0;
    for(i=0;i<ndim;i++){
        idx[i]=0;
        idx_current[i]=0;
    }
    size=1;
    for(i=0;i<ndim;i++){
        size*=m;
        size*=nmax+1;
    }
    x_glob=(double **)malloc(size*sizeof(double *));
    c_glob=(double *)malloc(size*sizeof(double));
    for(counter=0;counter<size;counter++){
        x_glob[counter]=(double *)malloc(ndim*sizeof(double));
        c_glob[counter]=1.0;
    }
    delta=1.0/(double)m;
    counter=0;
    while(!carry){
        //Initial array setup;
        for(i=0;i<ndim;i++){
            a1[i]=delta*idx[i]*(b[i]-a[i])+a[i];
            b1[i]=(idx[i]+1)*delta*(b[i]-a[i])+a[i];
        }
        carry_current=0;
        for(i=0;i<ndim;i++){
            idx_current[i]=0;//Initial setup;
        }
        while(!carry_current){
            for(i=0;i<ndim;i++){
                x[i]=ksi[idx_current[i]]*(b1[i]-a1[i])/2.0+(a1[i]+b1[i])/2.0;
                x_glob[counter][i]=x[i];
            }
            //      eta=f(x,serviceData);
            for(i=0;i<ndim;i++){
                c_glob[counter]*=an[idx_current[i]]*(b1[i]-a1[i])/2.0;
            }
            //      rv+=eta;
            counter++;
            carry_current=increment(idx_current,ndim,nmax);
        }
        carry=increment(idx,ndim,m-1);
    }
    //Fill the structures;
    for(i=0;i<nproc;i++){
        tparams[i].f=&f;
        tparams[i].x_glob=x_glob;
        tparams[i].c_glob=c_glob;
        tparams[i].serviceData=serviceData;
        tparams[i].ndim=ndim;
        tparams[i].size=size;
        tparams[i].nproc=nproc;
        tparams[i].tid=i;
        pthread_create(&tids[i],0,pt_proc,(void *)&tparams[i]);
    }
    for(i=0;i<nproc;i++){
        pthread_join(tids[i],(void **)&retval[i]);
    }
    rv=0;
    for(i=0;i<nproc;i++){
        rv+=*retval[i];
    }
    free(a1);
    free(b1);
    free(idx);
    free(idx_current);
    for(counter=0;counter<size;counter++){
        free(x_glob[counter]);
    }
    free(x_glob);
    free(c_glob);
    free(tparams);
    free(tids);
    free(x);
    free(retval);
    return rv;
}

long double GaussIntegrateParallelLong(std::function<long double(long double [],void *)> f, void *serviceData, int ndim, long double a[], long double b[], int m, unsigned int nproc){
    pthread_t *tids;
    pt_params_long *tparams;
    int *idx,*idx_current;
    int carry,carry_current;
    long double rv,eta;
    long double *a1,*b1;
    long double *x;
    long double **x_glob,*c_glob;
    long double **retval;
    long long int size;
    long double delta;
    int i,j;
    //define Gauss rule;
    static const long double ksi[15]={-0.9879925180204854,-0.937273392400706,-0.8482065834104272,-0.7244177313601701,-0.5709721726085388,-0.3941513470775634,-0.2011940939974345,0,0.2011940939974345,0.3941513470775634,0.5709721726085388,0.7244177313601701,0.8482065834104272,0.937273392400706,0.9879925180204854};
    static const long double an[15]={0.03075324199611749,0.07036604748810815,0.107159220467172,0.1395706779261543,0.1662692058169939,0.1861610000155622,0.1984314853271116,0.2025782419255613,0.1984314853271116,0.1861610000155622,0.1662692058169939,0.1395706779261543,0.107159220467172,0.07036604748810815,0.03075324199611749};
    int nmax=14;
    long long int counter=0;
    idx_current=(int *)malloc(ndim*sizeof(int));
    x=(long double *)malloc(ndim*sizeof(long double));
    idx=(int *)malloc(ndim*sizeof(int));
    a1=(long double *)malloc(ndim*sizeof(long double));
    b1=(long double *)malloc(ndim*sizeof(long double));
    retval=(long double **)malloc(nproc*sizeof(long double *));
    tids=(pthread_t *)malloc(nproc*sizeof(pthread_t));
    tparams=(pt_params_long *)malloc(nproc*sizeof(pt_params_long));
    carry=0;
    carry_current=0;
    rv=0;
    for(i=0;i<ndim;i++){
        idx[i]=0;
        idx_current[i]=0;
    }
    size=1;
    for(i=0;i<ndim;i++){
        size*=m;
        size*=nmax+1;
    }
    x_glob=(long double **)malloc(size*sizeof(long double *));
    c_glob=(long double *)malloc(size*sizeof(long double));
    for(counter=0;counter<size;counter++){
        x_glob[counter]=(long double *)malloc(ndim*sizeof(long double));
        c_glob[counter]=1.0;
    }
    delta=1.0/(long double)m;
    counter=0;
    while(!carry){
        //Initial array setup;
        for(i=0;i<ndim;i++){
            a1[i]=delta*idx[i]*(b[i]-a[i])+a[i];
            b1[i]=(idx[i]+1)*delta*(b[i]-a[i])+a[i];
        }
        carry_current=0;
        for(i=0;i<ndim;i++){
            idx_current[i]=0;//Initial setup;
        }
        while(!carry_current){
            for(i=0;i<ndim;i++){
                x[i]=ksi[idx_current[i]]*(b1[i]-a1[i])/2.0+(a1[i]+b1[i])/2.0;
                x_glob[counter][i]=x[i];
            }
            //      eta=f(x,serviceData);
            for(i=0;i<ndim;i++){
                c_glob[counter]*=an[idx_current[i]]*(b1[i]-a1[i])/2.0;
            }
            //      rv+=eta;
            counter++;
            carry_current=increment(idx_current,ndim,nmax);
        }
        carry=increment(idx,ndim,m-1);
    }
    //Fill the structures;
    for(i=0;i<nproc;i++){
        tparams[i].f=&f;
        tparams[i].x_glob=x_glob;
        tparams[i].c_glob=c_glob;
        tparams[i].serviceData=serviceData;
        tparams[i].ndim=ndim;
        tparams[i].size=size;
        tparams[i].nproc=nproc;
        tparams[i].tid=i;
        pthread_create(&tids[i],0,pt_proc_long,(void *)&tparams[i]);
    }
    for(i=0;i<nproc;i++){
        pthread_join(tids[i],(void **)&retval[i]);
    }
    rv=0;
    for(i=0;i<nproc;i++){
        rv+=*retval[i];
    }
    free(a1);
    free(b1);
    free(idx);
    free(idx_current);
    for(counter=0;counter<size;counter++){
        free(x_glob[counter]);
    }
    free(x_glob);
    free(c_glob);
    free(tparams);
    free(tids);
    free(x);
    free(retval);
    return rv;
}



double GaussIntegrateParallelEpsilon(std::function<double(double [],void *)> f, void *serviceData, int ndim, double a[], double b[], double epsrel, unsigned int nproc){
    int m;
    bool stop = false;
    double current,old,delta;
    m = 1;
    while(!stop){
        old = GaussIntegrateParallel(f,serviceData,ndim,a,b,m,nproc);
        m*=2;
        current = GaussIntegrateParallel(f,serviceData,ndim,a,b,m,nproc);
        delta = std::fabs(old-current)/std::fabs(current);
        if(delta<epsrel)
        {
            stop = true;
        }
        if(m>2048){
            stop = true;//Not convergence;
        }
    }
    return current;
}

//Realize cubature of Gauss Rule using pthreads.
double_complex_t ZGaussIntegrateParallel(std::function<double_complex_t(double [],void *)> f,void *serviceData,int ndim,double a[],double b[],int m,unsigned int nproc){
    pthread_t *tids;
    pt_params_z *tparams;
    int *idx,*idx_current;
    int carry,carry_current;
    double_complex_t rv,eta;
    double *a1,*b1;
    double *x;
    double **x_glob,*c_glob;
    double_complex_t **retval;
    long long int size;
    double delta;
    int i,j;
    //define Gauss rule;
    static const double ksi[15]={-0.9879925180204854,-0.937273392400706,-0.8482065834104272,-0.7244177313601701,-0.5709721726085388,-0.3941513470775634,-0.2011940939974345,0,0.2011940939974345,0.3941513470775634,0.5709721726085388,0.7244177313601701,0.8482065834104272,0.937273392400706,0.9879925180204854};
    static const double an[15]={0.03075324199611749,0.07036604748810815,0.107159220467172,0.1395706779261543,0.1662692058169939,0.1861610000155622,0.1984314853271116,0.2025782419255613,0.1984314853271116,0.1861610000155622,0.1662692058169939,0.1395706779261543,0.107159220467172,0.07036604748810815,0.03075324199611749};
    int nmax=14;
    long long int counter=0;
    idx_current=(int *)malloc(ndim*sizeof(int));
    x=(double *)malloc(ndim*sizeof(double));
    idx=(int *)malloc(ndim*sizeof(int));
    a1=(double *)malloc(ndim*sizeof(double));
    b1=(double *)malloc(ndim*sizeof(double));
    retval=(double_complex_t **)malloc(nproc*sizeof(double_complex_t*));
    tids=(pthread_t *)malloc(nproc*sizeof(pthread_t));
    tparams=(pt_params_z *)malloc(nproc*sizeof(pt_params_z));
    carry=0;
    carry_current=0;
    rv=0;
    for(i=0;i<ndim;i++){
        idx[i]=0;
        idx_current[i]=0;
    }
    size=1;
    for(i=0;i<ndim;i++){
        size*=m;
        size*=nmax+1;
    }
    x_glob=(double **)malloc(size*sizeof(double *));
    c_glob=(double *)malloc(size*sizeof(double));
    for(counter=0;counter<size;counter++){
        x_glob[counter]=(double *)malloc(ndim*sizeof(double));
        c_glob[counter]=1.0;
    }
    delta=1.0/(double)m;
    counter=0;
    while(!carry){
        //Initial array setup;
        for(i=0;i<ndim;i++){
            a1[i]=delta*idx[i]*(b[i]-a[i])+a[i];
            b1[i]=(idx[i]+1)*delta*(b[i]-a[i])+a[i];
        }
        carry_current=0;
        for(i=0;i<ndim;i++){
            idx_current[i]=0;//Initial setup;
        }
        while(!carry_current){
            for(i=0;i<ndim;i++){
                x[i]=ksi[idx_current[i]]*(b1[i]-a1[i])/2.0+(a1[i]+b1[i])/2.0;
                x_glob[counter][i]=x[i];
            }
            //      eta=f(x,serviceData);
            for(i=0;i<ndim;i++){
                c_glob[counter]*=an[idx_current[i]]*(b1[i]-a1[i])/2.0;
            }
            //      rv+=eta;
            counter++;
            carry_current=increment(idx_current,ndim,nmax);
        }
        carry=increment(idx,ndim,m-1);
    }
    //Fill the structures;
    for(i=0;i<nproc;i++){
        tparams[i].f=&f;
        tparams[i].x_glob=x_glob;
        tparams[i].c_glob=c_glob;
        tparams[i].serviceData=serviceData;
        tparams[i].ndim=ndim;
        tparams[i].size=size;
        tparams[i].nproc=nproc;
        tparams[i].tid=i;
        pthread_create(&tids[i],0,pt_proc_z,(void *)&tparams[i]);
    }
    for(i=0;i<nproc;i++){
        pthread_join(tids[i],(void **)&retval[i]);
    }
    rv=0;
    for(i=0;i<nproc;i++){
        rv+=*retval[i];
    }
    free(a1);
    free(b1);
    free(idx);
    free(idx_current);
    free(x_glob);
    free(c_glob);
    free(tparams);
    free(tids);
    free(x);
    return rv;
}

double_complex_t ZGaussIntegrateParallelEpsilon(std::function<double_complex_t(double [],void *)> f, void *serviceData, int ndim, double a[], double b[], double epsrel, unsigned int nproc){
    int m;
    bool stop = false;
    double_complex_t current,old;
    double delta;
    m = 1;
    while(!stop){
        old = ZGaussIntegrateParallel(f,serviceData,ndim,a,b,m,nproc);
        m*=2;
        current = ZGaussIntegrateParallel(f,serviceData,ndim,a,b,m,nproc);
        delta = cabs(current-old)/cabs(current);
        if(delta<epsrel)
        {
            stop = true;
        }
        if(m>2048){
            stop = true;//Not convergence;
        }
    }
    return current;
}

int  integrand_cuba(unsigned ndim, const double *x, void *sd,
                   unsigned fdim, double *fval)
{
    cuba_t *p;
    p = (cuba_t *)sd;
    *fval=p->f((double *)x,p->sd);
    return 0;
}
int  integrand_cuba_z(unsigned ndim, const double *x, void *sd,
                   unsigned fdim, double *fval)
{
    zcuba_t *p;
    p = (zcuba_t *)sd;
    fval[0]=creal(p->f((double *)x,p->sd));
    fval[1]=cimag(p->f((double *)x,p->sd));
    return 0;
}
//Now -- introduce adaptive Genz Malik cubature;
double GaussIntegrateCuba(std::function<double(double [],void *)> f, void *serviceData, int ndim, double a[], double b[],double epsabs, double epsrel,int maxeval)
{
    cuba_t p;
    double rv,err;
    p.f = f;
    p.sd = serviceData;
    p.a = a;
    p.b = b;
    p.ndim = ndim;
    hcubature(1,integrand_cuba,(void *)&p,ndim,a,b,maxeval,epsabs,epsrel,ERROR_INDIVIDUAL,&rv,&err);
    return rv;
}

double_complex_t ZGaussIntegrateCuba(std::function<double_complex_t(double [],void *)> f, void *serviceData, int ndim, double a[], double b[],double epsabs, double epsrel,int maxeval)
{
    zcuba_t p;
    double rv[2],err;
    p.f = f;
    p.sd = serviceData;
    p.a = a;
    p.b = b;
    p.ndim = ndim;
    hcubature(2,integrand_cuba_z,(void *)&p,ndim,a,b,maxeval,epsabs,epsrel,ERROR_INDIVIDUAL,rv,&err);
    return rv[0]+I*rv[1];
}

int IsNaN(double x){
    int rv;
    rv=0;
    if(x>0.0){
        rv=0;
    }else{
        if(x<=0.0){
            rv=0;
        }else{
            rv=1;
        }
    }
    return rv;
}

int IsInf(double x){
    return 1.0/x==0;
}

int sign(double x){
    if(x>0.0){
        return 1;
    }
    if(x<0.0){
        return -1;
    }
    if(x==0.0){
        return 0;
    }
    return 0;
}

int KronekerDelta(int l1,int l2){
    if(l1==l2){
        return 1;
    }
    return 0;
}

//TODO:fix n<0;
double LegendreP(int n,double x){
    int i;
    double arr[3];
    arr[0]=1.0;
    arr[1]=x;
    for(i=1;i<n;i++){
        arr[2]=((2.0*i+1.0)*x*arr[1]-i*arr[0])/(i+1.0);
        arr[0]=arr[1];
        arr[1]=arr[2];
    }
    if(n==0){
        return arr[0];
    }else{
        return arr[1];
    }
}

double LegendreQ(int n,double x){
    int i;
    double arr[3];
    arr[0]=1.0;
    arr[1]=x;
    for(i=1;i<n;i++){
        arr[2]=((2.0*i+1.0)*x*arr[1]-i*arr[0])/(i+1.0);
        arr[0]=arr[1];
        arr[1]=arr[2];
    }
    if(n==0){
        return arr[0];
    }else{
        return arr[1];
    }
}

double DLegendreP(int n,double x){
    return (LegendreP(n+1,x)-x*LegendreP(n,x))*(n+1)/(x*x-1.0);
}

double FindZero(double (*f)(double,void *),double a,double b,void *serviceData,int * errcode){
    double z1,z2,z3,znew,zold;
    double f1,f2,f3;
    double epsilon,er_rel;
    double qj,Aj,Bj,Cj;
    int maxiter,cnt,cont_iter;
    maxiter=100;
    cnt=0;
    *errcode=0;
    epsilon=1e-18;
    cont_iter=1;
    if((!IsNaN(f(a,serviceData)))&&(!IsNaN(f(b,serviceData)))){
        if(f(a,serviceData)*f(b,serviceData)>=0.0){
            if(f(a,serviceData)==0.0){
                return a;
            }
            if(f(b,serviceData)==0.0){
                return b;
            }
            *errcode=1;
            return 0.0/0.0;
        }else{
            z1=a;
            z2=b;
            z3=(a+b)/2.0;
            zold=z1;
            for(cnt=0;cnt<maxiter&&cont_iter;cnt++){
                f1=f(z1,serviceData);
                f2=f(z2,serviceData);
                f3=f(z3,serviceData);
                qj=(z3-z2)/(z2-z1);
                Aj=qj*f3-qj*(1.0+qj)*f2+f1*qj*qj;
                Bj=(2.0*qj+1.0)*f3-(1.0+qj)*(1.0+qj)*f2+f1*qj*qj;
                Cj=f3*(1.0+qj);
                znew=z3-sign(Bj)*(z3-z2)*(2.0*Cj)/(fabs(Bj)+sqrt(Bj*Bj-4.0*Aj*Cj));
                if(IsNaN(znew)){
                    cont_iter=0;
                }else{
                    if(f1*f3>0.0){
                        z1=z2;
                        z2=z3;
                        z3=znew;
                    }else{
                        z2=z3;
                        z3=znew;
                    }
                }
            }
            er_rel=fabs((znew-zold)/znew);
            if(er_rel<epsilon){
                cont_iter=0;
            }
            zold=znew;
        }
        return z2;
    }else{
        *errcode=1;
        return 0.0/0.0;
    }
}

//Find i-th zero of function f;
double FindNZero(double(*f)(double,void *),double x0,double x1,void *serviceData,double sep,int i,int * nf){
    double x;
    double a,b,eps;
    int cnt;
    *nf=0;
    a=x0;
    b=x0+sep;
    cnt=0;
    eps=1e-10;
    if(x0>x1){
        *nf=1;
        printf("Error. Check bounds. Exit\n");
        return 0.0/0.0;
    }
    if(i==0){
        if(f(x0,serviceData)==0.0){
            return x0;
        }else{
            *nf=1;
            return 0.0/0.0;
        }
    }
    while((cnt<i)&&(a<x1)){
        if(b>=x1){
            b=x1;
            if(f(a,serviceData)*f(b,serviceData)<0 || f(b,serviceData)==0){
                cnt++;
            }
        }else{
            if(f(a,serviceData)*f(b,serviceData)<0 || f(b,serviceData)==0){
                cnt++;
            }
            b=b+sep;
        }
        a=a+sep;
    }
    if(cnt<i){
        *nf=1;
        return 0.0/0.0;
    }else{
        if(a<x1){
            b=b-sep;
        }
        a=a-sep;
        return FindZero(f,a,b,serviceData,nf);
    }
}

long double FindZeroLong(long double (*f)(long double,void *),long double a,long double b,void *serviceData,int * errcode){
    long double z1,z2,z3,znew,zold;
    long double f1,f2,f3;
    long double epsilon,er_rel;
    long double qj,Aj,Bj,Cj;
    int maxiter,cnt,cont_iter;
    maxiter=100;
    cnt=0;
    *errcode=0;
    epsilon=1e-18;
    cont_iter=1;
    if((!IsNaN(f(a,serviceData)))&&(!IsNaN(f(b,serviceData)))){
        if(f(a,serviceData)*f(b,serviceData)>=0.0){
            if(f(a,serviceData)==0.0){
                return a;
            }
            if(f(b,serviceData)==0.0){
                return b;
            }
            *errcode=1;
            return 0.0/0.0;
        }else{
            z1=a;
            z2=b;
            z3=(a+b)/2.0;
            zold=z1;
            for(cnt=0;cnt<maxiter&&cont_iter;cnt++){
                f1=f(z1,serviceData);
                f2=f(z2,serviceData);
                f3=f(z3,serviceData);
                qj=(z3-z2)/(z2-z1);
                Aj=qj*f3-qj*(1.0+qj)*f2+f1*qj*qj;
                Bj=(2.0*qj+1.0)*f3-(1.0+qj)*(1.0+qj)*f2+f1*qj*qj;
                Cj=f3*(1.0+qj);
                znew=z3-sign(Bj)*(z3-z2)*(2.0*Cj)/(fabs(Bj)+sqrt(Bj*Bj-4.0*Aj*Cj));
                if(IsNaN(znew)){
                    cont_iter=0;
                }else{
                    if(f1*f3>0.0){
                        z1=z2;
                        z2=z3;
                        z3=znew;
                    }else{
                        z2=z3;
                        z3=znew;
                    }
                }
            }
            er_rel=fabs((znew-zold)/znew);
            if(er_rel<epsilon){
                cont_iter=0;
            }
            zold=znew;
        }
        return z2;
    }else{
        *errcode=1;
        return 0.0/0.0;
    }
}

//Find i-th zero of function f;
long double FindNZeroLong(long double(*f)(long double,void *),long double x0,long double x1,void *serviceData,long double sep,int i,int * nf){
    long double x;
    long double a,b,eps;
    int cnt;
    *nf=0;
    a=x0;
    b=x0+sep;
    cnt=0;
    eps=1e-10;
    if(x0>x1){
        *nf=1;
        printf("Error. Check bounds. Exit\n");
        return 0.0/0.0;
    }
    if(i==0){
        if(f(x0,serviceData)==0.0){
            return x0;
        }else{
            *nf=1;
            return 0.0/0.0;
        }
    }
    while((cnt<i)&&(a<x1)){
        if(b>=x1){
            b=x1;
            if(f(a,serviceData)*f(b,serviceData)<0 || f(b,serviceData)==0){
                cnt++;
            }
        }else{
            if(f(a,serviceData)*f(b,serviceData)<0 || f(b,serviceData)==0){
                cnt++;
            }
            b=b+sep;
        }
        a=a+sep;
    }
    if(cnt<i){
        *nf=1;
        return 0.0/0.0;
    }else{
        if(a<x1){
            b=b-sep;
        }
        a=a-sep;
        return FindZeroLong(f,a,b,serviceData,nf);
    }
}

void dump_v(std::string &filename,std::vector<double> v)
{
    FILE *f = fopen(filename.c_str(),"w");
    for(int i=0;i<v.size();i++)
    {
        printf("v[%d]=%.16lg\n",i,v[i]);
    }
    fclose(f);
}


double LaguerreL(int n,int m,double x){
    if(n<0){
        return 0;
    }
    if(n==0){
        return 1;
    }
    if(n==1){
        return m+1-x;
    }
    return ((2*(n-1)+m+1-x)*(LaguerreL(n-1,m,x))-(n-1+m)*LaguerreL(n-2,m,x))/(n);
}

long double LaguerreLa(int n,long double a,double x){
    if(n<0){
        return 0;
    }
    if(n==0){
        return 1;
    }
    if(n==1){
        return a+1-x;
    }
    return ((2*(n-1)+a+1-x)*(LaguerreLa(n-1,a,x))-(n-1+a)*LaguerreL(n-2,a,x))/(n);
}


//Input: a[m,n];b[n,k]
//Output: r[m,k]
void MatrixMatrixMultiply(double *r,double *a,double *b,int m,int n,int k){
    int i,j,cnt;
    double sum;
    for(i=0;i<m;i++){
        for(j=0;j<k;j++){
            sum=0;
            for(cnt=0;cnt<n;cnt++){
                sum+=a[i*n+cnt]*b[cnt*k+j];
            }
            r[i*k+j]=sum;
        }
    }
    return;
}

double SQR(double x){
    return x*x;
}

double pythag_m ( double a, double b )
{
    double a1,b1;
    a1=fabs(a);
    b1=fabs(b);
    if(a1>b1){
        return a1*sqrt(1.0+SQR(b1/a1));
    }else{
        return (b1==0.0? 0.0 : b1*sqrt(1.0+SQR(a1/b1)));
    }
}

void QR_decompose_rotation(double *a,double *q,double *r,int n){
    int i,j,k;
    int idxi,idxj;
    double c,s,sr,ri,rj,qi,qj,tmp;
    for(i=0;i<n*n;i++){
        r[i]=a[i];
        q[i]=0;
    }
    for(i=0;i<n;i++){
        q[i*n+i]=1.0;
    }
    for(j=0;j<n;j++){
        for(i=j+1;i<n;i++){
            sr=pythag_m(r[j*n+j],r[i*n+j]);
            if(sr==0.0){
                c=1.0;
                s=0.0;
            }else{
                c=r[j*n+j]/sr;
                s=r[i*n+j]/sr;
            }
            //Do rotation Tij
            for(k=0;k<n;k++){
                idxi=i*n+k;
                idxj=j*n+k;
                ri=r[idxi]*c-r[idxj]*s;
                rj=r[idxi]*s+r[idxj]*c;
                qi=q[idxi]*c-q[idxj]*s;
                qj=q[idxi]*s+q[idxj]*c;
                r[idxi]=ri;
                r[idxj]=rj;
                q[idxi]=qi;
                q[idxj]=qj;
            }
        }
    }
    //Do final transpose
    for(i=0;i<n;i++){
        for(j=i;j<n;j++){
            tmp=q[i*n+j];
            q[i*n+j]=q[j*n+i];
            q[j*n+i]=tmp;
        }
    }
    return;
}

//Решатель СЛАУ
void QR_solve(int N,double *q,double *r,double *b,double *x){
    int i,j;
    double *bn;
    double sum;
    bn=(double *)malloc(N*sizeof(double));
    for(i=0;i<N;i++){
        bn[i]=0;
        for(j=0;j<N;j++){
            bn[i]+=q[j*N+i]*b[j];//bn=Qt*b;
        }
    }
    for(i=N-1;i>=0;i--){
        sum=0;
        for(j=i+1;j<N;j++){
            sum+=r[i*N+j]*x[j];
        }
        x[i]=(bn[i]-sum)/r[i*N+i];
    }
    free(bn);
    return;
}

void QR_decompose_reflection(double *a,double *q,double *r,int n){
    int i,j,k;
    double *v;
    double norm,dot1,dot2,tmp;
    v=(double *)malloc(n*sizeof(double));
    for(i=0;i<n*n;i++){
        r[i]=a[i];
        q[i]=0;
    }
    for(i=0;i<n;i++){
        q[i*n+i]=1.0;
    }
    for(j=0;j<n-1;j++){
        norm=0;
        for(i=0;i<n;i++){
            v[i]=0;
            if(i>=j){
                v[i]=r[i*n+j];
            }
            norm+=v[i]*v[i];
        }
        norm=sqrt(norm);
        if(v[j]>0){
            v[j]=v[j]+norm;
        }else{
            v[j]=v[j]-norm;
        }
        norm=0;
        for(i=0;i<n;i++){
            norm+=v[i]*v[i];
        }
        norm=sqrt(norm);
        if(norm!=0.0){
            for(i=0;i<n;i++){
                v[i]=v[i]/norm;
            }
        }
        for(k=0;k<n;k++){
            dot1=0;
            dot2=0;
            for(i=0;i<n;i++){
                dot1+=r[i*n+k]*v[i];
                dot2+=q[i*n+k]*v[i];
            }
            dot1=dot1*2.0;
            dot2=dot2*2.0;
            for(i=0;i<n;i++){
                r[i*n+k]=r[i*n+k]-v[i]*dot1;
                q[i*n+k]=q[i*n+k]-v[i]*dot2;
            }
        }
    }
    //Do final transpose
    for(i=0;i<n;i++){
        for(j=i;j<n;j++){
            tmp=q[i*n+j];
            q[i*n+j]=q[j*n+i];
            q[j*n+i]=tmp;
        }
    }
    free(v);
}

void printMatrix(double *a,char * title,int m,int n){
    int i,j;
    if(title!=NULL){
        printf("%s\n",title);
    }
    for(i=0;i<m;i++){
        for(j=0;j<n;j++){
            printf("%20.13lg ",a[i*n+j]);
        }
        printf("\n");
    }
}

void printMatrixZ(double_complex_t *a,char * title,int m,int n){
    int i,j;
    if(title!=NULL){
        printf("%s\n",title);
    }
    for(i=0;i<m;i++){
        for(j=0;j<n;j++){
            printf(" (%20.13lg + I %20.13lg) ",creal(a[i*n+j]),cimag(a[i*n+j]));
        }
        printf("\n");
    }
}

//Realisation of Runge-Kutta method of 4-th order
//Integrand calling convention:
//F(neq,t,y[],rv[]);
int rk4_step(void (*F)(int,double,double [],double[],void *),int neq,double y[],double t,double dt,void *serviceData){
    int i;
    double *k1,*k2,*k3,*k4;
    double *y1,*rv;
    //Allocating memory
    k1=(double *)malloc(neq*sizeof(double));
    k2=(double *)malloc(neq*sizeof(double));
    k3=(double *)malloc(neq*sizeof(double));
    k4=(double *)malloc(neq*sizeof(double));
    y1=(double *)malloc(neq*sizeof(double));
    rv=(double *)malloc(neq*sizeof(double));
    F(neq,t,y,rv,serviceData);
    for(i=0;i<neq;i++){
        k1[i]=dt*rv[i];
        y1[i]=y[i]+0.5*k1[i];
    }
    F(neq,t+0.5*dt,y1,rv,serviceData);
    for(i=0;i<neq;i++){
        k2[i]=dt*rv[i];
        y1[i]=y[i]+0.5*k2[i];
    }
    F(neq,t+0.5*dt,y1,rv,serviceData);
    for(i=0;i<neq;i++){
        k3[i]=dt*rv[i];
        y1[i]=y[i]+k3[i];
    }
    F(neq,t+dt,y1,rv,serviceData);
    for(i=0;i<neq;i++){
        k4[i]=dt*rv[i];
        y[i]=y[i]+(1.0/6.0)*(k1[i]+2.0*k2[i]+2.0*k3[i]+k4[i]);
    }
    free(k1);
    free(k2);
    free(k3);
    free(k4);
    free(y1);
    free(rv);
}

int rk4(void (*F)(int,double,double [],double[],void *),int neq,double y[],double t0,double dt,int nsteps,void *serviceData){
    double t;
    int i;
    t=t0;
    for(i=0;i<nsteps;i++){
        rk4_step(F,neq,y,t,dt,serviceData);
        t+=dt;
    }
}

int rk5_step(void (*F)(int,double,double [],double[],void *),int neq,double y[],double t,double dt,void *serviceData){
    int i;
    double *k1,*k2,*k3,*k4,*k5,*k6;
    double *y1,*rv;
    //Allocating memory
    k1=(double *)malloc(neq*sizeof(double));
    k2=(double *)malloc(neq*sizeof(double));
    k3=(double *)malloc(neq*sizeof(double));
    k4=(double *)malloc(neq*sizeof(double));
    k5=(double *)malloc(neq*sizeof(double));
    k6=(double *)malloc(neq*sizeof(double));
    y1=(double *)malloc(neq*sizeof(double));
    rv=(double *)malloc(neq*sizeof(double));
    F(neq,t,y,rv,serviceData);
    for(i=0;i<neq;i++){
        k1[i]=dt*rv[i];
        y1[i]=y[i]+0.25*k1[i];
    }
    F(neq,t+0.25*dt,y1,rv,serviceData);
    for(i=0;i<neq;i++){
        k2[i]=dt*rv[i];
        y1[i]=y[i]+(3.0/32.0)*k1[i]+(9.0/32.0)*k2[i];
    }
    F(neq,t+(3.0/8.0)*dt,y1,rv,serviceData);
    for(i=0;i<neq;i++){
        k3[i]=dt*rv[i];
        y1[i]=y[i]+(1932.0/2197.0)*k1[i]-(7200.0/2197.0)*k2[i]+(7296.0/2197.0)*k3[i];
    }
    F(neq,t+(12.0/13.0)*dt,y1,rv,serviceData);///i am here
    for(i=0;i<neq;i++){
        k4[i]=dt*rv[i];
        y1[i]=y[i]+(439.0/216.0)*k1[i]-8.0*k2[i]+(3680.0/513.0)*k3[i]-(845.0/4104.0)*k4[i];
    }
    F(neq,t+dt,y1,rv,serviceData);
    for(i=0;i<neq;i++){
        k5[i]=dt*rv[i];
        y1[i]=y[i]-(8.0/27.0)*k1[i]+2.0*k2[i]-(3544.0/2565.0)*k3[i]+(1859.0/4104.0)*k4[i]-(11.0/40.0)*k5[i];
    }
    F(neq,t+0.5*dt,y1,rv,serviceData);
    for(i=0;i<neq;i++){
        k6[i]=dt*rv[i];
        y[i]=y[i]+(16.0/135.0)*k1[i]+(6656.0/12825.0)*k3[i]+(28561.0/56430.0)*k4[i]-(9.0/50.0)*k5[i]+(2.0/55.0)*k6[i];
    }
    free(k1);
    free(k2);
    free(k3);
    free(k4);
    free(k5);
    free(k6);
    free(y1);
    free(rv);
}

int rk5(void (*F)(int,double,double [],double[],void *),int neq,double y[],double t0,double dt,int nsteps,void *serviceData){
    double t;
    int i;
    t=t0;
    for(i=0;i<nsteps;i++){
        rk5_step(F,neq,y,t,dt,serviceData);
        t+=dt;
    }
}

//Runge-Kutta in complex area
//Realisation of Runge-Kutta method of 4-th order
//Integrand calling convention:
//F(neq,t,y[],rv[]);
int zrk4_step(void (*F)(int,double,double_complex_t [],double_complex_t [],void *),int neq,double_complex_t y[],double t,double dt,void *serviceData){
    int i;
    double_complex_t *k1,*k2,*k3,*k4;
    double_complex_t *y1,*rv;
    //Allocating memory
    k1=(double_complex_t *)malloc(neq*sizeof(double_complex_t));
    k2=(double_complex_t *)malloc(neq*sizeof(double_complex_t));
    k3=(double_complex_t *)malloc(neq*sizeof(double_complex_t));
    k4=(double_complex_t *)malloc(neq*sizeof(double_complex_t));
    y1=(double_complex_t *)malloc(neq*sizeof(double_complex_t));
    rv=(double_complex_t *)malloc(neq*sizeof(double_complex_t));
    F(neq,t,y,rv,serviceData);
    for(i=0;i<neq;i++){
        k1[i]=dt*rv[i];
        y1[i]=y[i]+0.5*k1[i];
    }
    F(neq,t+0.5*dt,y1,rv,serviceData);
    for(i=0;i<neq;i++){
        k2[i]=dt*rv[i];
        y1[i]=y[i]+0.5*k2[i];
    }
    F(neq,t+0.5*dt,y1,rv,serviceData);
    for(i=0;i<neq;i++){
        k3[i]=dt*rv[i];
        y1[i]=y[i]+k3[i];
    }
    F(neq,t+dt,y1,rv,serviceData);
    for(i=0;i<neq;i++){
        k4[i]=dt*rv[i];
        y[i]=y[i]+(1.0/6.0)*(k1[i]+2.0*k2[i]+2.0*k3[i]+k4[i]);
    }
    free(k1);
    free(k2);
    free(k3);
    free(k4);
    free(y1);
    free(rv);
}

int zrk4(void (*F)(int,double,double_complex_t [],double_complex_t[],void *),int neq,double_complex_t y[],double t0,double dt,int nsteps,void *serviceData){
    double t;
    int i;
    t=t0;
    for(i=0;i<nsteps;i++){
        zrk4_step(F,neq,y,t,dt,serviceData);
        t+=dt;
    }
}

int zrk5_step(void (*F)(int,double,double_complex_t [],double_complex_t[],void *),int neq,double_complex_t y[],double t,double dt,void *serviceData){
    int i;
    double_complex_t *k1,*k2,*k3,*k4,*k5,*k6;
    double_complex_t *y1,*rv;
    //Allocating memory
    k1=(double_complex_t *)malloc(neq*sizeof(double_complex_t));
    k2=(double_complex_t *)malloc(neq*sizeof(double_complex_t));
    k3=(double_complex_t *)malloc(neq*sizeof(double_complex_t));
    k4=(double_complex_t *)malloc(neq*sizeof(double_complex_t));
    k5=(double_complex_t *)malloc(neq*sizeof(double_complex_t));
    k6=(double_complex_t *)malloc(neq*sizeof(double_complex_t));
    y1=(double_complex_t *)malloc(neq*sizeof(double_complex_t));
    rv=(double_complex_t *)malloc(neq*sizeof(double_complex_t));
    F(neq,t,y,rv,serviceData);
    for(i=0;i<neq;i++){
        k1[i]=dt*rv[i];
        y1[i]=y[i]+0.25*k1[i];
    }
    F(neq,t+0.25*dt,y1,rv,serviceData);
    for(i=0;i<neq;i++){
        k2[i]=dt*rv[i];
        y1[i]=y[i]+(3.0/32.0)*k1[i]+(9.0/32.0)*k2[i];
    }
    F(neq,t+(3.0/8.0)*dt,y1,rv,serviceData);
    for(i=0;i<neq;i++){
        k3[i]=dt*rv[i];
        y1[i]=y[i]+(1932.0/2197.0)*k1[i]-(7200.0/2197.0)*k2[i]+(7296.0/2197.0)*k3[i];
    }
    F(neq,t+(12.0/13.0)*dt,y1,rv,serviceData);///i am here
    for(i=0;i<neq;i++){
        k4[i]=dt*rv[i];
        y1[i]=y[i]+(439.0/216.0)*k1[i]-8.0*k2[i]+(3680.0/513.0)*k3[i]-(845.0/4104.0)*k4[i];
    }
    F(neq,t+dt,y1,rv,serviceData);
    for(i=0;i<neq;i++){
        k5[i]=dt*rv[i];
        y1[i]=y[i]-(8.0/27.0)*k1[i]+2.0*k2[i]-(3544.0/2565.0)*k3[i]+(1859.0/4104.0)*k4[i]-(11.0/40.0)*k5[i];
    }
    F(neq,t+0.5*dt,y1,rv,serviceData);
    for(i=0;i<neq;i++){
        k6[i]=dt*rv[i];
        y[i]=y[i]+(16.0/135.0)*k1[i]+(6656.0/12825.0)*k3[i]+(28561.0/56430.0)*k4[i]-(9.0/50.0)*k5[i]+(2.0/55.0)*k6[i];
    }
    free(k1);
    free(k2);
    free(k3);
    free(k4);
    free(k5);
    free(k6);
    free(y1);
    free(rv);
}

int zrk5(void (*F)(int, double, double_complex_t[], double_complex_t[], void *), int neq, double_complex_t y[], double t0, double dt, int nsteps, void *serviceData){
    double t;
    int i;
    t=t0;
    for(i=0;i<nsteps;i++){
        zrk5_step(F,neq,y,t,dt,serviceData);
        t+=dt;
    }
}

//Realisation of Felberg method 7,8 order with accuracy control.
//Integrand calling convention:
//F(neq,t,y[],rv[]);
int fel78(void (*F)(int,double,double [],double[],void *),int neq,double y[],double t0,double t,void *serviceData){
    int i,j,k;
    const int sz = 13;
    const int N = 50000;
    const double epsilon = 1e-12;//Обеспечиваем решения с хорошей точностью.
    double **ki;
    double *delta;
    double step,q;
    double alpha[]={0,2.0/27.0,1.0/9.0,1.0/6.0,5.0/12.0,1.0/2.0,5.0/6.0,1.0/6.0,2.0/3.0,1.0/3.0,1.0,0.0,1.0};
    double p7i[]={41.0/840.0,0,0,0,0,34.0/105.0,9.0/35.0,9.0/35.0,9.0/280.0,9.0/280.0,41.0/840.0,0,0};
    double p8i[]={0,0,0,0,0,34.0/105.0,9.0/35.0,9.0/35.0,9.0/280.0,9.0/280.0,0,41.0/840.0,41.0/840.0};
    double **beta;
    double *y1,*rv;
    //Allocating memory
    ki = (double **)malloc(sz*sizeof(double *));
    for(i=0;i<sz;i++)
    {
        ki[i] = (double *)malloc(neq*sizeof(double));
    }
    beta = (double **)malloc(sz*sizeof(double *));
    for(i=0;i<sz;i++)
    {
        beta[i] = (double *)malloc(sz*sizeof(double));
        for(j=0;j<sz;j++)
        {
            beta[i][j]=0.0;//Init;
        }
    }
    y1=(double *)malloc(neq*sizeof(double));
    rv=(double *)malloc(neq*sizeof(double));
    delta = (double *)malloc(neq*sizeof(double));
    //Теперь надо заполнить данными.
    beta[1][0] = 2.0/27.0;
    beta[2][0] = 1.0/36.0;
    beta[2][1] = 1.0/12.0;
    beta[3][0] = 1.0/24.0;
    beta[3][1] = 0;
    beta[3][2] = 1.0/8.0;
    beta[4][0] = 5.0/12.0;
    beta[4][1] = 0;
    beta[4][2] = -25.0/16.0;
    beta[4][3] = 25.0/16.0;
    beta[5][0] = 1.0/20.0;
    beta[5][1] = 0;
    beta[5][2] = 0;
    beta[5][3] = 1.0/4.0;
    beta[5][4] = 1.0/5.0;
    beta[6][0] = -25.0/108.0;
    beta[6][1] = 0;
    beta[6][2] = 0;
    beta[6][3] = 125.0/108.0;
    beta[6][4] = -65.0/27.0;
    beta[6][5] = 125.0/54.0;
    beta[7][0] = 31.0/300.0;
    beta[7][1] = 0;
    beta[7][2] = 0;
    beta[7][3] = 0;
    beta[7][4] = 61.0/225.0;
    beta[7][5] = -2.0/9.0;
    beta[7][6] = 13.0/900.0;
    beta[8][0] = 2.0;
    beta[8][1] = 0;
    beta[8][2] = 0;
    beta[8][3] = 23.0/108.0;
    beta[8][4] = 704.0/45.0;
    beta[8][5] = -107.0/9.0;
    beta[8][6] = 67.0/90.0;
    beta[8][7] = 3.0;
    beta[9][0] = -91.0/108.0;
    beta[9][1] = 0;
    beta[9][2] = 0;
    beta[9][3] = 23.0/108.0;
    beta[9][4] = -976.0/135.0;
    beta[9][5] = 311.0/54.0;
    beta[9][6] = -19.0/60.0;
    beta[9][7] = 17.0/6.0;
    beta[9][8] = -1.0/12.0;
    beta[10][0] = 2383.0/4100.0;
    beta[10][1] = 0;
    beta[10][2] = 0;
    beta[10][3] = -341.0/164.0;
    beta[10][4] = 4496.0/1025.0;
    beta[10][5] = -301.0/82.0;
    beta[10][6] = 2133.0/4100.0;
    beta[10][7] = 45.0/82.0;
    beta[10][8] = 45.0/164.0;
    beta[10][9] = 18.0/41.0;
    beta[11][0] = 3.0/205.0;
    beta[11][1] = 0;
    beta[11][2] = 0;
    beta[11][3] = 0;
    beta[11][4] = 0;
    beta[11][5] = -6.0/41.0;
    beta[11][6] = -3.0/205.0;
    beta[11][7] = -3.0/41.0;
    beta[11][8] = 3.0/41.0;
    beta[11][9] = 6.0/41.0;
    beta[11][10] = 0;
    beta[12][0] = -1777.0/4100.0;
    beta[12][1] = 0;
    beta[12][2] = 0;
    beta[12][3] = -341.0/164.0;
    beta[12][4] = 4496.0/1025.0;
    beta[12][5] = -289.0/82.0;
    beta[12][6] = -2193.0/4100.0;
    beta[12][7] = 51.0/82.0;
    beta[12][8] = 33.0/164.0;
    beta[12][9] = 12.0/41.0;
    beta[12][10] = 0;
    beta[12][11] = 1.0;
    //Теперь запускаем итерационную процедуру.
    step = (t-t0)/N;//Начальное, какое-то значение шага, которое будет уточнятся.
    double tn;
    bool prot = false;
    tn = t0;
    while((step > 0? tn<t: tn>t))
    {
        q = 0.0;
        bool flag = true;
        while(flag)
        {
            if(q >=1.0)
            {
                flag = false;
            }
            for(i=0;i<sz;i++)
            {
                //Находим ki.
                for(j=0;j<i;j++)
                {
                    for(k=0;k<neq;k++)
                    {
                        y1[k] = y[k] + beta[i][j] * ki[j][k];
                    }
                }
                F(neq,tn + alpha[i] * step,y1,rv,serviceData);
                for(k=0;k<neq;k++)
                {
                    ki[i][k] = step * rv[k];
                }
            }
            //Находим точность
            for(k=0;k<neq;k++)
            {
                delta[k] = 0;
                for(i=0;i<sz;i++)
                {
                    delta[k] += (p8i[i]-p7i[i])*ki[i][k];
                }
            }
            double norm = 0;
            for(k=0;k<neq;k++)
            {
                norm +=fabs(delta[k])*fabs(delta[k]);
            }
            norm = sqrt(norm);
            if(prot)
            {
                flag = false;//Больше никаких итераций.
            }
            if(flag)
            {
                q = pow(epsilon / norm,1.0/8.0);
                step = q*step;
            }
        }
        for(k=0;k<neq;k++)
        {
            for(i=0;i<sz;i++)
            {
                y[k] += p8i[i]*ki[i][k];
            }
        }
        if((step>0?(tn + step)>t:(tn + step)<t))
        {
            step = t-tn;//И здесь надо поставить запрет изменятся шагу.
            prot = true;
        }
        tn+=step;
    }
    //Free memory
    for(i=0;i<sz;i++)
    {
        free(ki[i]);
    }
    free(ki);
    for(i=0;i<sz;i++)
    {
        free(beta[i]);
    }
    free(beta);
    free(y1);
    free(rv);
    free(delta);
    return 0;
}


//Realisation of Felberg method 7,8 order with accuracy control.
//Integrand calling convention:
//F(neq,t,y[],rv[]);
int zfel78(void (*F)(int,double,double_complex_t [],double_complex_t[],void *),int neq,double_complex_t y[],double t0,double t,void *serviceData){
    int i,j,k;
    const int sz = 13;
    const int N = 50000;
    const double epsilon = 1e-12;//Обеспечиваем решения с хорошей точностью.
    double_complex_t **ki;
    double_complex_t *delta;
    double step,q;
    double alpha[]={0,2.0/27.0,1.0/9.0,1.0/6.0,5.0/12.0,1.0/2.0,5.0/6.0,1.0/6.0,2.0/3.0,1.0/3.0,1.0,0.0,1.0};
    double p7i[]={41.0/840.0,0,0,0,0,34.0/105.0,9.0/35.0,9.0/35.0,9.0/280.0,9.0/280.0,41.0/840.0,0,0};
    double p8i[]={0,0,0,0,0,34.0/105.0,9.0/35.0,9.0/35.0,9.0/280.0,9.0/280.0,0,41.0/840.0,41.0/840.0};
    double **beta;
    double_complex_t *y1,*rv;
    //Allocating memory
    ki = (double_complex_t **)malloc(sz*sizeof(double_complex_t *));
    for(i=0;i<sz;i++)
    {
        ki[i] = (double_complex_t *)malloc(neq*sizeof(double_complex_t));
    }
    beta = (double **)malloc(sz*sizeof(double *));
    for(i=0;i<sz;i++)
    {
        beta[i] = (double *)malloc(sz*sizeof(double));
        for(j=0;j<sz;j++)
        {
            beta[i][j]=0.0;//Init;
        }
    }
    y1=(double_complex_t *)malloc(neq*sizeof(double_complex_t));
    rv=(double_complex_t *)malloc(neq*sizeof(double_complex_t));
    delta = (double_complex_t *)malloc(neq*sizeof(double_complex_t));
    //Теперь надо заполнить данными.
    beta[1][0] = 2.0/27.0;
    beta[2][0] = 1.0/36.0;
    beta[2][1] = 1.0/12.0;
    beta[3][0] = 1.0/24.0;
    beta[3][1] = 0;
    beta[3][2] = 1.0/8.0;
    beta[4][0] = 5.0/12.0;
    beta[4][1] = 0;
    beta[4][2] = -25.0/16.0;
    beta[4][3] = 25.0/16.0;
    beta[5][0] = 1.0/20.0;
    beta[5][1] = 0;
    beta[5][2] = 0;
    beta[5][3] = 1.0/4.0;
    beta[5][4] = 1.0/5.0;
    beta[6][0] = -25.0/108.0;
    beta[6][1] = 0;
    beta[6][2] = 0;
    beta[6][3] = 125.0/108.0;
    beta[6][4] = -65.0/27.0;
    beta[6][5] = 125.0/54.0;
    beta[7][0] = 31.0/300.0;
    beta[7][1] = 0;
    beta[7][2] = 0;
    beta[7][3] = 0;
    beta[7][4] = 61.0/225.0;
    beta[7][5] = -2.0/9.0;
    beta[7][6] = 13.0/900.0;
    beta[8][0] = 2.0;
    beta[8][1] = 0;
    beta[8][2] = 0;
    beta[8][3] = 23.0/108.0;
    beta[8][4] = 704.0/45.0;
    beta[8][5] = -107.0/9.0;
    beta[8][6] = 67.0/90.0;
    beta[8][7] = 3.0;
    beta[9][0] = -91.0/108.0;
    beta[9][1] = 0;
    beta[9][2] = 0;
    beta[9][3] = 23.0/108.0;
    beta[9][4] = -976.0/135.0;
    beta[9][5] = 311.0/54.0;
    beta[9][6] = -19.0/60.0;
    beta[9][7] = 17.0/6.0;
    beta[9][8] = -1.0/12.0;
    beta[10][0] = 2383.0/4100.0;
    beta[10][1] = 0;
    beta[10][2] = 0;
    beta[10][3] = -341.0/164.0;
    beta[10][4] = 4496.0/1025.0;
    beta[10][5] = -301.0/82.0;
    beta[10][6] = 2133.0/4100.0;
    beta[10][7] = 45.0/82.0;
    beta[10][8] = 45.0/164.0;
    beta[10][9] = 18.0/41.0;
    beta[11][0] = 3.0/205.0;
    beta[11][1] = 0;
    beta[11][2] = 0;
    beta[11][3] = 0;
    beta[11][4] = 0;
    beta[11][5] = -6.0/41.0;
    beta[11][6] = -3.0/205.0;
    beta[11][7] = -3.0/41.0;
    beta[11][8] = 3.0/41.0;
    beta[11][9] = 6.0/41.0;
    beta[11][10] = 0;
    beta[12][0] = -1777.0/4100.0;
    beta[12][1] = 0;
    beta[12][2] = 0;
    beta[12][3] = -341.0/164.0;
    beta[12][4] = 4496.0/1025.0;
    beta[12][5] = -289.0/82.0;
    beta[12][6] = -2193.0/4100.0;
    beta[12][7] = 51.0/82.0;
    beta[12][8] = 33.0/164.0;
    beta[12][9] = 12.0/41.0;
    beta[12][10] = 0;
    beta[12][11] = 1.0;
    //Теперь запускаем итерационную процедуру.
    step = (t-t0)/N;//Начальное, какое-то значение шага, которое будет уточнятся.
    double tn;
    bool prot = false;
    tn = t0;
    while((step > 0? tn<t: tn>t))
    {
        q = 0.0;
        bool flag = true;
        while(flag)
        {
            if(q >=1.0)
            {
                flag = false;
            }
            for(i=0;i<sz;i++)
            {
                //Находим ki.
                for(j=0;j<i;j++)
                {
                    for(k=0;k<neq;k++)
                    {
                        y1[k] = y[k] + beta[i][j] * ki[j][k];
                    }
                }
                F(neq,tn + alpha[i] * step,y1,rv,serviceData);
                for(k=0;k<neq;k++)
                {
                    ki[i][k] = step * rv[k];
                }
            }
            //Находим точность
            for(k=0;k<neq;k++)
            {
                delta[k] = 0;
                for(i=0;i<sz;i++)
                {
                    delta[k] += (p8i[i]-p7i[i])*ki[i][k];
                }
            }
            double norm = 0;
            for(k=0;k<neq;k++)
            {
                norm +=cabs(delta[k])*cabs(delta[k]);
            }
            norm = sqrt(norm);
            if(prot)
            {
                flag = false;//Больше никаких итераций.
            }
            if(flag)
            {
                q = pow(epsilon / norm,1.0/8.0);
                step = q*step;
            }
        }
        for(k=0;k<neq;k++)
        {
            for(i=0;i<sz;i++)
            {
                y[k] += p8i[i]*ki[i][k];
            }
        }
        if((step>0?(tn + step)>t:(tn + step)<t))
        {
            step = t-tn;//И здесь надо поставить запрет изменятся шагу.
            prot = true;
        }
        tn+=step;
    }
    //Free memory
    for(i=0;i<sz;i++)
    {
        free(ki[i]);
    }
    free(ki);
    for(i=0;i<sz;i++)
    {
        free(beta[i]);
    }
    free(beta);
    free(y1);
    free(rv);
    free(delta);
}


double ipow(double x,int k){
    int i;
    double res;
    res=1.0;
    if(k>0){
        for(i=0;i<k;i++){
            res*=x;
        }
    }
    if(k<0){
        for(i=0;i<-k;i++){
            res/=x;
        }
    }
    return res;
}

double ifact(int k){
    double rv;
    int i;
    rv=1.0;
    for(i=1;i<=k;i++){
        rv*=i;
    }
    return rv;
}

void print_complex(char * msg,double_complex_t z){
    printf("%s (%.16lg %.16lg)\n",msg,creal(z),cimag(z));
}

void print_complex_vector(char * msg,int n,double_complex_t *z){
    int i;
    printf("%s\n",msg);
    for(i=0;i<n;i++){
        printf("(%.16lg %.16lg)\n",creal(z[i]),cimag(z[i]));
    }
}

void print_complex_matrix(char * msg,int n,double_complex_t *z){
    int i,j;
    printf("%s\n",msg);
    for(i=0;i<n;i++){
        for(j=0;j<n;j++){
            printf("(%.16lg %.16lg) ",creal(z[i*n+j]),cimag(z[i*n+j]));
        }
        printf("\n");
    }
}


//Use the continued fraction representation
double_complex_t _zgamcontinued(double_complex_t z,double_complex_t a,double_complex_t b,double_complex_t c,double_complex_t d,int deep){
    double_complex_t term;
    if(deep>0){
        return a-b/(c+d/_zgamcontinued(z,a+2,b+z,c+2,d+z,deep-1));
    }else{
        return a;
    }
}


double_complex_t ZGammaIncomplete(double s,double_complex_t z){
    double_complex_t a,b,c,d;
    double_complex_t term;
    int deep;
    a=s;
    b=s*z;
    c=s+1;
    d=z;
    deep=150;
    return cpow(z,s)*cexp(-z)/_zgamcontinued(z,a,b,c,d,deep);
}

double_complex_t _zexpintegrale(double_complex_t z,double_complex_t a,double_complex_t b,double_complex_t c,double_complex_t d,int deep){
    double_complex_t term;
    if(deep>0){
        return a+b/(c+d/_zexpintegrale(z,a,b+1,c,d+1,deep-1));
    }else{
        return a;
    }
}


double_complex_t ZExpIntegralE(double nu,double_complex_t z){
    double_complex_t a,b,c,d;
    double_complex_t term;
    int deep;
    a=z;
    b=nu;
    c=1;
    d=1;
    deep=150;
    return cexp(-z)/_zexpintegrale(z,a,b,c,d,deep);
}

double min(double a,double b){
    if(a<b){
        return a;
    }
    return b;
}

double max(double a,double b){
    if(a<b){
        return b;
    }
    return a;
}

double fmax(double a,double b){
    if(a<b){
        if(b>0)
        {
            return b;
        }
        else
        {
            return a;
        }
    }
    if(a>0)
    {
        return a;
    }
    else
    {
        return b;
    }
}

double ClebschGordan(double j1,double m1,double j2,double m2,double j3,double m3){
    double rval,a,b,n,xmin,xmax;
    int nmin,nmax,i,v;
    a=0;
    b=0;
    if((m1+m2!=m3)||(fabs(m1)>j1)||(fabs(m2)>j2)||(fabs(m3)>j3)){
        rval=0;
    }else{
        if((fabs(j1-j2)<=j3)&&(fabs(j1+j2)>=j3)){
            xmin=0;
            xmin=max(xmin,j2-j3-m1);
            xmin=max(xmin,j1-j3+m2);
            xmax=min(j2+m2,j1+j2-j3);
            xmax=min(xmax,j1-m1);
            nmin=round(xmin);
            nmax=round(xmax);
            a=sqrt((2.0*j3+1.0)*Gamma(j1+j2-j3+1)*Gamma(j3+j1-j2+1)*Gamma(j3+j2-j1+1)/Gamma(j1+j2+j3+2));
            b=0;
            for(i=nmin;i<=nmax;i++){
                n=i;
                b=b+(ipow(-1,i))*sqrt(Gamma(j1+m1+1)*Gamma(j1-m1+1)*Gamma(j2+m2+1)*Gamma(j2-m2+1)*Gamma(j3+m3+1)*Gamma(j3-m3+1))/(Gamma(n+1)*Gamma(j1+j2-j3-i+1)*Gamma(j1-m1-i+1)*Gamma(j2+m2-i+1)*Gamma(j3-j2+m1+i+1)*Gamma(j3-j1-m2+i+1));
            }
            rval=a*b;
        }else{
            rval=0;
        }
    }
    return rval;
}


//Implementation of Borwein Arc Bessel technique.
//TODO.

//Use external implementation


double dbesselj(double nu, double z){
    return creal(besselj(nu,z));
}

double dbessely(double nu, double z){
    return creal(bessely(nu,z));
}

double dbesseli(double nu, double z){
    return creal(besseli(nu,z));
}

double dbesselk(double nu, double z){
    return creal(besselk(nu,z));
}



double_complex_t besselj(double nu, double_complex_t z){
    int kode=1;
    int n=1;
    double zr=creal(z);
    double zi=cimag(z);
    std::vector<int> nz={0};
    std::vector<double> cyr={0},cyi={0};
    std::vector<int> ierr={0};
    double_complex_t res;
    if(nu<0){
        return (2*nu+2)*besselj(nu+1,z)/z-besselj(nu+2,z);
    }
    zbesj(zr,zi,nu,kode,n,cyr,cyi,nz,ierr);
    if(ierr[0]!=0){
        printf("error!\n");
    }
    res=cyr[0]+I*cyi[0];
    return res;
}

double_complex_t bessely(double nu, double_complex_t z){
    int kode=1;
    int n=1;
    double zr=creal(z);
    double zi=cimag(z);
    std::vector<int> nz={0};
    std::vector<double> cyr={0},cyi={0};
    std::vector<double> wrkr={0},wrki={0};
    double_complex_t res;
    std::vector<int> ierr={0};
    if(nu<0){
        return (2*nu+2)*bessely(nu+1,z)/z-bessely(nu+2,z);
    }
    zbesy(zr,zi,nu,kode,n,cyr,cyi,nz,wrkr,wrki,ierr);
    if(ierr[0]!=0){
        printf("error!\n");
    }
    res=cyr[0]+I*cyi[0];
    return res;
}

double_complex_t besseli(double nu, double_complex_t z){
    int kode=1;
    int n=1;
    double zr=creal(z);
    double zi=cimag(z);
    std::vector<int> nz={0};
    std::vector<double> cyr={0},cyi={0};
    double_complex_t res;
    std::vector<int> ierr={0};
    if(nu<0){
        return (2*nu+2)*besseli(nu+1,z)/z+besseli(nu+2,z);
    }
    zbesi(zr,zi,nu,kode,n,cyr,cyi,nz,ierr);
    if(ierr[0]!=0){
        printf("error!\n");
    }
    res=cyr[0]+I*cyi[0];
    return res;
}

double_complex_t besselk(double nu, double_complex_t z){
    int kode=1;
    int n=1;
    double zr=creal(z);
    double zi=cimag(z);
    std::vector<int> nz={0};
    std::vector<double> cyr={0},cyi={0};
    double_complex_t res;
    std::vector<int> ierr={0};
    if(nu<0){
        return -(2*nu+2)*besselk(nu+1,z)/z+besselk(nu+2,z);
    }
    zbesk(zr,zi,nu,kode,n,cyr,cyi,nz,ierr);
    if(ierr[0]!=0){
        printf("error!\n");
    }
    res=cyr[0]+I*cyi[0];
    return res;
}

//Spherical Bessel Functions
double sj(int l,double x){
    return sqrt(pi/(2*x))*dbesselj(l+0.5,x);
}
double sy(int l,double x){
    return sqrt(pi/(2*x))*dbessely(l+0.5,x);
}
double si(int l,double x){
    return sqrt(pi/(2*x))*dbesseli(l+0.5,x);
}
double sk(int l,double x){
    return sqrt(pi/(2*x))*dbesselk(l+0.5,x);
}


//Add new code
double Pochgammer(int n,double alpha){
    int i;
    double prod=1;
    for(i=0;i<n;i++){
        prod*=(alpha+i);
    }
    return prod;
}

double_complex_t ZPochgammer(int n,double_complex_t alpha){
    int i;
    double_complex_t prod=1;
    for(i=0;i<n;i++){
        prod*=(alpha+i);
    }
    return prod;
}

double Hypergeometric1F1(double a,double c,double z){
    double sum=0;
    double eps=1e-16;
    double term=1;
    int i,maxiter;
    maxiter=10000;
    /*if(z>10.0)
    {
        return Hypergeometric1F1(c-a,c,-z)*exp(z);//Use Kummer formula
    }*/
    for(i=0;i<maxiter && fabs(term) >fabs(sum)*eps;i++){
        term=ipow(z,i)*Pochgammer(i,a)/Pochgammer(i,c)/ifact(i);
        sum+=term;
    }
    return sum;
}

double_complex_t ZHypergeometric1F1(double_complex_t a,double_complex_t c,double z){
    double_complex_t sum=0;
    double eps=1e-16;
    double_complex_t term=1;
    int i,maxiter;
    maxiter=100;
    if(z<0.0)
    {
        return ZHypergeometric1F1(c-a,c,-z)*exp(z);//Use Kummer formula
    }
    for(i=0;i<maxiter && cabs(term) >cabs(sum)*eps;i++){
        term=ipow(z,i)*ZPochgammer(i,a)/ZPochgammer(i,c)/ifact(i);
        sum+=term;
    }
    return sum;
}

/**
 * Вычисляет вырожденную гипергеометрическую функцию Куммера 1F1(a; b; z)
 *
 * @param a комплексный параметр
 * @param b комплексный параметр (не должен быть нулём или отрицательным целым,
 *           если только a не является целым отрицательным с |a| <= |b|)
 * @param z вещественный аргумент
 * @param eps относительная точность (по умолчанию 1e-12)
 * @param max_iter максимальное число членов ряда (по умолчанию 10000)
 * @return значение функции M(a,b,z)
 * @throws std::domain_error если b — неположительное целое, а a не удовлетворяет условию
 */
long_double_complex_t hypergeometric_1F1(
    long_double_complex_t a,
    long_double_complex_t b,
    long double z,
    long double eps,
    int max_iter)
{
    // Проверка на тривиальный случай z = 0
    if (z == 0.0) {
        return 1.0;
    }

    // Вспомогательная функция для проверки, является ли число целым (с допуском)
    auto is_integer = [](long double x) {
        const long double eps_int = 1e-12;
        return std::abs(x - std::round(x)) < eps_int;
    };

    // Обработка особых случаев: b — неположительное целое
    const long double eps_imag = 1e-12;
    if (std::abs(cimag(b)) < eps_imag && creal(b) <= 0.0 && is_integer(creal(b))) {
        int m = static_cast<int>(std::round(creal(b))); // b = -m, m >= 0
        if (m == 0) {
            throw std::domain_error("hypergeometric_1F1: b = 0 вызывает полюс");
        }
        // Проверяем, является ли a целым отрицательным
        if (std::abs(cimag(a)) < eps_imag && creal(a) <= 0.0 && is_integer(creal(a))) {
            int k = static_cast<int>(std::round(-creal(a))); // a = -k, k >= 0
            if (k <= m) {
                // Ряд обрывается, суммируем полином степени k
                long_double_complex_t sum = 1.0;
                long_double_complex_t term = 1.0;
                for (int n = 1; n <= k; ++n) {
                    term *= (a + static_cast<long double>(n-1)) / (b + static_cast<long double>(n-1)) * (z / n);
                    sum += term;
                }
                return sum;
            } else {
                throw std::domain_error("hypergeometric_1F1: b отрицательное целое, a не удовлетворяет условию обрыва ряда");
            }
        } else {
            throw std::domain_error("hypergeometric_1F1: b отрицательное целое, a не является целым отрицательным");
        }
    }

    // Для отрицательных z используем преобразование Куммера для улучшения сходимости
    // M(a,b,z) = exp(z) * M(b-a, b, -z)
    if (z < 0.0) {
        long_double_complex_t res = hypergeometric_1F1(b - a, b, -z, eps, max_iter);
        return std::exp(z) * res;
    }

    // Суммирование гипергеометрического ряда для z >= 0
    long_double_complex_t sum = 1.0;
    long_double_complex_t term = 1.0;
    int n = 0;
    while (n < max_iter) {
        ++n;
        // term_{n} = term_{n-1} * (a+n-1)/(b+n-1) * z/n
        term *= (a + static_cast<long double>(n-1)) / (b + static_cast<long double>(n-1)) * (z / n);
        sum += term;
        // Проверка сходимости
        if (cabsl(term) <= eps * cabsl(sum) && n > 5) {
            break;
        }
    }
    if (n == max_iter) {
        // Достигнуто максимальное число итераций — возможно, требуется больше членов
        // Вместо исключения можно вернуть текущее значение с предупреждением
        // Для простоты генерируем исключение
        throw std::runtime_error("hypergeometric_1F1: превышено максимальное число итераций");
    }
    return sum;
}


double to_a_f4(double *x_,void *sd)
{
    double u,rv;
    double a,b,c1,c2,z1,z2;
    appellf4_params *p;
    p = (appellf4_params *)sd;
    u=x_[0];
    a = p->a;
    b = p->b;
    c1 = p->c1;
    c2 = p->c2;
    z1 = p->z1;
    z2 = p->z2;
    rv = pow(u,b-1)*creal(Hypergeometric2F1(a/2.0,a/2.0+1.0/2.0,c1,4*u*u*z1*z2/((-1+(z1+z2)*u)*(-1+(z1+z2)*u))))/(pow(1-u,b-c1+1.0)*pow(1-(z1+z2)*u,a));
    return rv;
}

/**
 * @brief AppellF4
 * Реализация функции Аппеля F4.
 * Без аналитического продолжения.
 * @param a
 * @param b
 * @param c1
 * @param c2
 * @param x
 * @param y
 * @return
 */
double AppellF4(double a,double b,double c1,double c2,double x,double y)
{
    assert(sqrt(fabs(x))+sqrt(fabs(y))<1.0);//Функция реализована только в круге сходимости.
    assert(c1==c2);//Только для таких.
    appellf4_params p;
    double rv, l, r;
    p.a = a;
    p.b = b;
    p.c1 = c1;
    p.c2 = c2;
    p.z1 = x;
    p.z2 = y;
    l = 0.0;
    r = 1.0;
    rv = (Gamma(c1)/(Gamma(b)*Gamma(c1-b)))*GaussIntegrateParallel(to_a_f4,(void *)&p,1,&l,&r,20,get_num_threads());
    return rv;
}

double EulerBeta(double x,double y){
    return Gamma(x)*Gamma(y)/Gamma(x+y);
}

double to_m(double *tp,void *sd){
    w_params *ptr;
    double t,lambda,mu,z;
    ptr=(w_params *)sd;
    t=*tp;
    lambda=ptr->lambda;
    mu=ptr->mu;
    z=ptr->z;
    return pow((1+t),mu-lambda-1.0/2.0)*pow((1-t),mu+lambda-1.0/2.0)*exp(z*t/2.0);
}

double WhittakerM(double lambda,double mu,double z){
    double result,a,b;
    w_params p;
    p.mu=mu;
    p.lambda=lambda;
    p.z=z;
    a=-1;
    b=1;
    result=pow(z,mu+1.0/2.0)/(pow(2.0,2*mu)*EulerBeta(mu+lambda+1.0/2.0,mu-lambda+1.0/2.0));
    result=result*GaussIntegrate(to_m,(void *)&p,1,&a,&b,1000);
    return result;
}

double_complex_t to_w(double *t_,void *sd){
    w_params_complex *ptr;
    double t,z;
    double_complex_t lambda,mu;
    ptr=(w_params_complex *)sd;
    t=*t_;
    lambda=ptr->lambda;
    mu=ptr->mu;
    z=ptr->z;
    return exp(-z*t)*cpow(t,mu-lambda-1.0/2.0)*cpow(1+t,mu+lambda-1.0/2.0);
}

double_complex_t WhittakerW(double_complex_t lambda,double_complex_t mu,double z)
{
    double_complex_t result,mu_c,lambda_c;
    std::vector<double_complex_t> retvals,mus,lambdas;
    double a1,a2,a3,a4,a5;
    w_params_complex p;
    p.lambda = lambda;
    p.mu = mu;
    p.z = z;
    a1 = 0.0;
    a2 = 1.0;
    a3 = 10.0;
    a4 = 100.0;
    a5 = 1000.0;
    mu_c = mu;
    lambda_c = lambda;
    int n = int(2.0-creal(mu_c-lambda_c));
    if(n<2)
        n=2;
    retvals.resize(n+1);
    mus.resize(n+1);
    lambdas.resize(n+1);
    for(int i=0;i<=n;i++)
    {
        mu_c = mu;
        lambda_c = lambda + i - n;
        p.lambda = lambda_c;
        p.mu = mu_c;
        p.z = z;
        if(i<=1)//Используем интегральное представление только в области хорошей сходимости.
        {
            result = exp(-z/2.0)*cpow(z,mu_c+1.0/2.0)/ZGamma(mu_c-lambda_c+1.0/2.0);
            result=result*(ZGaussIntegrateParallel(to_w,(void *)&p,1,&a1,&a2,1000,get_num_threads())+\
                               ZGaussIntegrateParallel(to_w,(void *)&p,1,&a2,&a3,1000,get_num_threads())+\
                               ZGaussIntegrateParallel(to_w,(void *)&p,1,&a3,&a4,1000,get_num_threads())+\
                               ZGaussIntegrateParallel(to_w,(void *)&p,1,&a4,&a5,1000,get_num_threads()));
            retvals[i] = result;
            mus[i] = mu_c;
            lambdas[i] = lambda_c;
        }
        else//Используем рекуррентное соотношение.
        {
            result = -(2*lambdas[i-1] - z)*retvals[i-1]-(lambdas[i-1]-mus[i-1]-1.0/2.0)*(lambdas[i-1]+mus[i-1]-1.0/2.0)*retvals[i-2];
            retvals[i] = result;
            mus[i] = mu_c;
            lambdas[i] = lambda_c;
        }
    }
    return result;
}


double_complex_t to_f2(double *z,void *sd){
    f2_params *p;
    double u,v;
    double alpha,beta1,beta2,gamma1,gamma2,x,y;
    double_complex_t result;
    double_complex_t ca,cb;
    p=(f2_params *)sd;
    alpha=p->alpha;
    beta1=p->beta1;
    beta2=p->beta2;
    gamma1=p->gamma1;
    gamma2=p->gamma2;
    x=p->x;
    y=p->y;
    u=z[0];
    v=z[1];
    result=cpow(u,beta1-1.0)*cpow(v,beta2-1.0)*cpow(1-u,gamma1-beta1-1)*cpow(1-v,gamma2-beta2-1)*cpow(1-u*x-v*y,-alpha);
    if(IsInf(creal(result))){
        result=0;
    }
    if(IsNaN(creal(cpow(1-u*x-v*y,-alpha)))){
        result=cpow(u,beta1-1.0)*cpow(v,beta2-1.0)*cpow(1-u,gamma1-beta1-1)*cpow(1-v,gamma2-beta2-1);//Hack;; 0^0=1;
    }
    return result;
}

double_complex_t HypergeometricF2(double alpha,double beta1,double beta2,double gamma1,double gamma2,double x,double y){
    f2_params p;
    double_complex_t result;
    double a[2];
    double b[2];
    p.alpha=alpha;
    p.beta1=beta1;
    p.beta2=beta2;
    p.gamma1=gamma1;
    p.gamma2=gamma2;
    p.x=x;
    p.y=y;
    a[0]=0.0;
    b[0]=1.0;
    a[1]=0.0;
    b[1]=1.0;
    result=Gamma(gamma1)*Gamma(gamma2)/(Gamma(beta1)*Gamma(beta2)*Gamma(gamma1-beta1)*Gamma(gamma2-beta2));
    result*=ZGaussIntegrate(to_f2,(void *)&p,2,a,b,10);
    return result;
}

//простой ряд с ограниченной сходимостью
double HypergeometricFA(int n,double alpha[],double beta [],double gamma [],double z[]){
    int *mi;
    int i,carry,m_max,ms;
    double prod,sum;
    mi=(int *)malloc(n*sizeof(int));
    for(i=0;i<n;i++){
        mi[i]=0;
    }
    carry=0;
    m_max=100;
    sum=0;
    while(!carry){
        prod=1;
        ms=0;
        for(i=0;i<n;i++){
            prod*=Pochgammer(mi[i],beta[i])*ipow(z[i],mi[i])/(Pochgammer(mi[i],gamma[i])*ifact(mi[i]));
            ms+=mi[i];
        }
        prod*=Pochgammer(ms,alpha[0]);
        sum+=prod;
        carry=increment(mi,n,m_max);
    }
    return sum;
}

void swap_double(double &a,double &b){
    double tmp;
    tmp=b;
    b=a;
    a=tmp;
}

void swap_vector_double(std::vector<double> &a,std::vector<double> &b){
    size_t size,i;
    double tmp;
    assert(a.size()==b.size());
    size=a.size();
    for(i=0;i<size;i++){
        tmp=b[i];
        b[i]=a[i];
        a[i]=tmp;
    }
}

void swap_vector_complex(std::vector<double_complex_t> &a,std::vector<double_complex_t> &b){
    size_t size,i;
    double_complex_t tmp;
    assert(a.size()==b.size());
    size=a.size();
    for(i=0;i<size;i++){
        tmp=b[i];
        b[i]=a[i];
        a[i]=tmp;
    }
}


//Original version of EigenSystem;
int EigenSystemOrig(int n,double *a,double_complex_t *w,double_complex_t *zc,int eigen_vectors){
    double *wr,*wi,*z;
    int i,j;
    wr=(double *)malloc(n*sizeof(double));
    wi=(double *)malloc(n*sizeof(double));
    z=(double *)malloc(n*n*sizeof(double));
    rg_elm(n,a,wr,wi,eigen_vectors,z);
    for(i=0;i<n;i++){
        w[i]=wr[i]+I*wi[i];
        if(eigen_vectors){
            for(j=0;j<n;j++){
                if(wi[i]>0){
                    zc[j*n+i]=z[j*n+i]+I*z[j*n+i+1];
                    zc[j*n+i+1]=z[j*n+i]-I*z[j*n+i+1];
                }
                if(wi[i]==0){
                    zc[j*n+i]=z[j*n+i];
                }//wi<0 --skip
            }
        }
    }
    free(wr);
    free(wi);
    free(z);
    return 0;
}
/*
int EigenSystem(int n,std::vector<std::vector<double>> a,std::vector<double_complex_t> &w,std::vector<std::vector<double_complex_t>> &zc,int eigen_vectors,int nproc){
    Eigen::setNbThreads(nproc);
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> M;
    M.resize(n,n);
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            M(i,j)=a[i][j];
        }
    }
    Eigen::EigenSolver<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>> es(M);
    Eigen::Matrix<std::complex<double>,Eigen::Dynamic,Eigen::Dynamic> ev=es.eigenvalues();
    w.clear();
    zc.clear();
    w.resize(n);
    std::vector<double_complex_t> cur;
    for(int i=0;i<n;i++){
        w[i]=ev(i,0).real()+I*ev(i,0).imag();
    }
    if(eigen_vectors){
        Eigen::Matrix<std::complex<double>,Eigen::Dynamic,Eigen::Dynamic> evec=es.eigenvectors();
        for(int j=0;j<n;j++){
            cur.clear();
            cur.resize(n);
            for(int i=0;i<n;i++){
                cur[i]=evec(i,j).real()+I*evec(i,j).imag();
            }
            zc.push_back(cur);
        }
    }
}
*/
int EigenSystemSym(int n,std::vector<std::vector<double>> a,std::vector<double_complex_t> &w,std::vector<std::vector<double_complex_t>> &zc,int eigen_vectors,int nproc){
    double *m,*v,*ww,*work;
    int lwork=16*n;
    int info;
    char c1='V';
    char c2='U';
    m=new double[n*n];
    ww=new double[n];
    work=new double[lwork];
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            m[i*n+j]=a[i][j];
        }
    }
    dsyev_(&c1,&c2,&n,m,&n,ww,work,&lwork,&info);
    if(info!=0){
        return -1;
    }
    w.clear();
    zc.clear();
    w.resize(n);
    std::vector<double_complex_t> cur;
    for(int i=0;i<n;i++){
        w[i]=ww[i];
    }
    if(eigen_vectors){
        for(int j=0;j<n;j++){
            cur.clear();
            cur.resize(n);
            for(int i=0;i<n;i++){
                cur[i]=m[j*n+i];
            }
            zc.push_back(cur);
        }
    }
    delete[]m;
}

int EigenSystemHerm(int n,std::vector<std::vector<double_complex_t>> a,std::vector<double_complex_t> &w,std::vector<std::vector<double_complex_t>> &zc,int eigenvectors,int nproc){
    char job='V';
    char up='U';
    double *ww,*rw;
    lapack_complex_t *m,*work;
    int info;
    m=new lapack_complex_t[n*n];
    ww=new double[n];
    int lwork=2*n;
    int rwork=3*n;
    work=new lapack_complex_t[lwork];
    rw=new double[rwork];
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            m[i*n+j].real=creal(a[i][j]);
            m[i*n+j].imag=cimag(a[i][j]);
        }
    }
    w.clear();
    zc.clear();
    w.resize(n);
    zheev_(&job,&up,&n,m,&n,ww,work,&lwork,rw,&info);
    std::vector<double_complex_t> cur;
    for(int i=0;i<n;i++){
        w[i]=ww[i];
    }
    if(eigenvectors){
        for(int j=0;j<n;j++){
            cur.clear();
            cur.resize(n);
            for(int i=0;i<n;i++){
                cur[i]=m[j*n+i].real+I*m[j*n+i].imag;
            }
            zc.push_back(cur);
        }
    }
    delete[]m;
    delete []work;
    delete []ww;
    delete []rw;
}

void zheevr(char jobz, char range, char uplo, int n, lapack_complex_t* a, int lda, double vl, double vu, int il, int iu, double abstol, double* w, lapack_complex_t* z, int ldz, int* info)
{
    int m;
    int lwork = -1;
    int liwork = -1;
    int lrwork = -1;
    int isuppz[2*n];
    lapack_complex_t small_work_doublecomplex[1];
    double small_work_double[1];
    int small_work_int[1];
    zheevr_(&jobz, &range, &uplo, &n, a, &lda, &vl, &vu, &il, &iu, &abstol, &m, w, z, &ldz, isuppz, small_work_doublecomplex, &lwork, small_work_double, &lrwork, small_work_int, &liwork, info);
    lwork = (int) small_work_doublecomplex[0].real;
    liwork = small_work_int[0];
    lrwork = (int) small_work_double[0];
    lapack_complex_t work[lwork];
    double rwork[lrwork];
    int iwork[liwork];
    zheevr_(&jobz, &range, &uplo, &n, a, &lda, &vl, &vu, &il, &iu, &abstol, &m, w, z, &ldz, isuppz, work, &lwork, rwork, &lrwork, iwork, &liwork, info);
}

int EigenSystemHerm2(int n,std::vector<std::vector<double_complex_t>> a,std::vector<double_complex_t> &w,std::vector<std::vector<double_complex_t>> &zc,int eigenvectors,int nproc){
    char job='V';
    char up='U';
    char range='A';
    double *ww,*rw;
    lapack_complex_t *m,*z,*work;
    int info;
    m=new lapack_complex_t[n*n];
    z=new lapack_complex_t[n*n];
    ww=new double[n];
    int lwork=32*n;
    int rwork=32*n;
    work=new lapack_complex_t[lwork];
    rw=new double[rwork];
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            m[i*n+j].real=creal(a[i][j]);
            m[i*n+j].imag=cimag(a[i][j]);
        }
    }
    w.clear();
    zc.clear();
    w.resize(n);
    const double abstol = std::numeric_limits<double>::min();
    zheevr(job,range,up,n,m,n,0.0,0.0,0,0,abstol,ww,z,n,&info);
    std::vector<double_complex_t> cur;
    for(int i=0;i<n;i++){
        w[i]=ww[i];
    }
    if(eigenvectors){
        for(int j=0;j<n;j++){
            cur.clear();
            cur.resize(n);
            for(int i=0;i<n;i++){
                cur[i]=z[j*n+i].real+I*z[j*n+i].imag;
            }
            zc.push_back(cur);
        }
    }
    delete []m;
    delete []work;
    delete []ww;
    delete []rw;
    delete []z;
}



double AiryAi(double x,int derivative){
    double zr,zi;
    std::vector<double> fr={0},fi={0};
    double zero=0.0;
    int id,kode;
    std::vector<int> err={0},nz={0};
    id=(derivative!=0?1:0);
    kode=1;
    zr=x;
    zi=0;
    zairy(zr,zi,id,kode,fr,fi,nz,err);
    if(err[0]){
        printf("ERROR\n");
    }
    return fr[0];
}

double AiryBi(double x,int derivative){
    double zr,zi;
    std::vector<double> fr={0},fi={0};
    double zero=0.0;
    int id,kode;
    std::vector<int> err={0};
    id=(derivative!=0?1:0);
    kode=1;
    zr=x;
    zi=0;
    zbiry(zr,zi,id,kode,fr,fi,err);
    if(err[0]){
        printf("ERROR\n");
    }
    return fr[0];
}

double Zeta(double s){
    int n,k,sign;
    double result=0;
    double addition;
    for(n=0;n<100;n++){
        addition=0;
        for(k=0;k<=n;k++){
            sign=(k%2==0?1:-1);
            addition+=(1.0/(pow(2,n+1)))*BinomCoeff(n,k)*sign/(pow(k+1,s));
        }
        result+=addition;
        if(fabs(addition/result)<1e-16){
            break;
        }
    }
    result=result/(1-pow(2,1-s));
    return result;
}

double_complex_t to_fourier(double *t,void *sd){
    fourier_param *p;
    double omega;
    p=(fourier_param *)sd;
    omega = p->omega;
    return p->F(*t,p->sd)*cexp(-I*omega*(*t));
}

double_complex_t Fourier(double_complex_t (*F)(double,void *sd),double omega,void *sd)
{
    double a,b;
    fourier_param p;
    p.F=F;
    p.omega=omega;
    p.sd=sd;
    a=-200.0;
    b=200.0;
    return ZGaussIntegrate(to_fourier,(void *)&p,1,&a,&b,200);
}

double PolyLog(double s,double z){
    double result,add,fact=1.0;
    int n;
    result=Gamma(1-s)*pow(log(1.0/z),s-1);
    for(n=0;n<100;n++){
        if(n>=1){
            fact*=n;
        }
        add=Zeta(s-n)*pow(log(z),n)/fact;
        result+=add;
        if(fabs(add/result)<1e-17){//Convergence
            break;
        }
    }
    return result;
}

void inverse(double* A, int N)
{
    int *IPIV = new int[N];
    int LWORK = N*N;
    double *WORK = new double[LWORK];
    int INFO;

    dgetrf_(&N,&N,A,&N,IPIV,&INFO);
    dgetri_(&N,A,&N,IPIV,WORK,&LWORK,&INFO);

    delete[] IPIV;
    delete[] WORK;
}

/**
 * @brief find_root
 * Процедура поиска корней системы нелинейных уравнений.
 * @param f
 * Тестируемая функция
 * @param x_init
 * Вектор начального приближения (размер -- число неизвестных.)
 * @param x_out
 * Вывод: решение системы
 * @param user_data
 * Пользовательские данные, которые передаются в функцию.
 * @param epsilon
 * Задаваемая погрешность
 * @param max_iter
 * Максимальное число итераций.
 * @return
 */

bool find_root(std::function<std::vector<double>(std::vector<double> &x,void *user_data)> f, const std::vector<double> &x_init, std::vector<double> &x_out, void *user_data, double epsilon, int max_iter)
{
    double dx = 0, f_norm = 0, x_norm = 0;
    double x_min = 0, x_max = 0;
    int iter = 0;
    bool flag = true;
    size_t num_dimensions;
    std::vector<double> x, x_shift, f_out;
    num_dimensions = x_init.size();
    std::vector<double> M;
    std::vector<double> V,VM;
    //    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> M;
    //    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> V;
    M.resize(num_dimensions*num_dimensions);
    V.resize(num_dimensions);
    VM.resize(num_dimensions);
    x.resize(num_dimensions);
    x_shift.resize(num_dimensions);
    f_out.resize(num_dimensions);
    for(int i = 0; i < num_dimensions; i ++)
    {
        x_shift[i] = x_init[i];
        x[i] = x_init[i];
    }
    //j -- function
    //i -- xi;
    //M(j,i);
    iter  = 0;
    x_min = fabs(x[0]);
    x_max = 0;
    for(int i = 0;i < num_dimensions;i ++)
    {
        x_min = (fabs(x[i]) < x_min ? fabs(x[i]) : x_min);
        x_max = (fabs(x[i]) > x_max ? fabs(x[i]) : x_max);
    }
    double x_mean = (x_min + x_max)/2.0;
    dx = (x_mean > 0 ? x_mean * epsilon : 0);
    do
    {
        for(int i = 0;i < num_dimensions;i ++)
        {
            x_shift[i] = x[i];
        }
        for(int i = 0;i < num_dimensions;i ++)
        {
            x_shift[i] = x[i] + dx;
            for(int j = 0;j < num_dimensions;j ++)
            {
                M[j*num_dimensions+i] = (f(x_shift, user_data)[j] - f(x,user_data)[j])/dx;
            }
            x_shift[i] = x[i];
        }
        inverse(M.data(),num_dimensions);
        for(int i = 0; i < num_dimensions; i ++)
        {
            V[i] = f(x,user_data)[i];
        }
        for(int i = 0; i < num_dimensions; i++)
        {
            VM[i] = 0;
            for(int j = 0; j < num_dimensions; j++)
            {
                VM[i] += M[i*num_dimensions+j]*V[j];
            }
        }
        for(int i = 0;i < num_dimensions;i++)
        {
            V[i] = VM[i];
        }
        //        V = M * V;
        for(int i = 0; i < num_dimensions; i ++)
        {
            x[i]=x[i]-V[i];
        }
        f_out=f(x,user_data);
        f_norm=0;
        x_norm=0;
        for(int i = 0; i < num_dimensions; i ++)
        {
            f_norm += f_out[i] * f_out[i];
        }
        f_norm = sqrt(f_norm);
        x_norm = 0;
        for(int i=0;i<V.size();i++)
        {
            x_norm += V[i]*V[i];
        }
        x_norm = sqrt(x_norm);
        if(f_norm < epsilon || x_norm < epsilon)
        {
            flag = false;//STOP;
        }
        iter++;
        if(iter >= max_iter)
        {
            //Convergence doesn't achieved
            x_out = x;
            return false;
        }
    }while(flag);
    x_out = x;
    return true;
}

bool find_minimum_1d(double(*f) (double &, void *), const double &x_init, double &x_out, void *user_data, double epsilon, int max_iter)
{
    double step;
    double a,b,c,x,y;//Концы интервала.
    double p,q,alpha;
    double s,fy,fx,fa,fb,fc,delta;
    double l,r;
    bool flag,direct;
    int cnt;
    step=epsilon;
    //    step=(fabs(x_init)>0? x_init*epsilon : epsilon);
    flag=true;
    //Итерации пока не определим интервал.
    //Далее -- автоматический выбор шага.
    l=x_init-step;
    r=x_init+step;
    while(f(l,user_data)==f(r,user_data))
    {
        step*=2;
        l=x_init-step;
        r=x_init+step;
    }
    step*=2.0;
    step*=5;
    c=x_init;
    b=c+step;
    fb=f(b,user_data);
    fc=f(c,user_data);
    cnt=0;
    direct=(fb<fc);
    if(direct){//DIRECT
        while(fb<fc){
            a=c;
            fa=fc;
            c=b;
            fc=fb;
            step*=2;
            b=c+step;
            fb=f(b,user_data);
        }
    }else{//INVERSE
        a=c-step;
        fa=f(a,user_data);
        while(fa<fc){
            b=c;
            fb=fc;
            c=a;
            fc=fa;
            step*=2;
            a=c-step;
            fa=f(a,user_data);
        }
    }
    //Есть интервал.
    if(b<a){
        c=a;
        a=b;
        b=c;
    }
    //Используем метод аппроксимации параболами.
    flag=true;
    x=(a+b)/2.0;
    fa=f(a,user_data);
    fb=f(b,user_data);
    fx=f(x,user_data);
    cnt=0;
    do{
        cnt++;
        p=(x-a)*(fb-fx);
        q=(b-x)*(fa-fx);
        s=(p+q);
        y=(p*(x+a)+q*(b+x))/(2*s);
        alpha=-s/((x*x-a*a)*(b-x)-(b*b-x*x)*(x-a));
        delta=fabs(x-y);
        if((y<=a) || (y>=b) || std::isnan(y) || std::isinf(y)){
            break;
        }
        fy=f(y,user_data);
        if(fy>fx){
            if(y<x){
                a=y;
                fa=fy;
            }else{
                b=y;
                fb=fy;
            }
        }else{
            if(y<x){
                b=x;
                fb=fx;
                x=y;
                fx=fy;
            }else{
                a=x;
                fa=fx;
                x=y;
                fx=fy;
            }
        }
        if(delta<epsilon){
            flag=false;
        }
        if(cnt>max_iter){
            return false;
        }
    }while(flag);
    if(fa<fx){
        x=a;
        fx=fa;
    }
    if(fb<fx){
        x=b;
        fx=fa;
    }
    x_out=x;
    return true;
}

/**
 * @brief find_minimum_descent
 * Поиск минимума методом градиентного спуска с линейным поиском Армихо.
 *
 * @param f          Целевая функция (принимает вектор и user_data, возвращает double)
 * @param x_init     Начальное приближение
 * @param x_out      Найденная точка минимума
 * @param user_data  Пользовательские данные
 * @param lambda_init Начальная длина шага (будет уменьшаться при необходимости)
 * @param epsilon     Точность по градиенту и изменению переменных
 * @param max_iter    Максимальное число итераций
 * @return true       Успех (сошлось), false – не сошлось
 */
bool find_minimum_descent(std::function<double (std::vector<double> &, void *)> f,
                          const std::vector<double> &x_init,
                          std::vector<double> &x_out,
                          void *user_data,
                          double lambda_init,
                          double epsilon,
                          int max_iter)
{
    const double dx_rel = 1e-7;          // относительный шаг для численного дифференцирования
    const double dx_abs = 1e-8;          // абсолютный минимальный шаг
    const double c_armijo = 1e-4;        // параметр Armijo (обычно 1e-4)
    const double lambda_reduce = 0.5;    // коэффициент уменьшения шага
    const double lambda_increase = 1.2;  // коэффициент увеличения шага (после успешной итерации)

    size_t n = x_init.size();
    std::vector<double> x = x_init;
    std::vector<double> grad(n);
    std::vector<double> x_new(n);
    std::vector<double> x_shift(n);

    double f_cur = f(x, user_data);
    if (std::isnan(f_cur) || std::isinf(f_cur)) {
        x_out = x;
        return false; // начальная точка невалидна
    }

    double lambda = lambda_init;
    int iter = 0;
    bool converged = false;

    while (iter < max_iter && !converged) {
        iter++;

        // ---------- 1. Вычисление градиента (центральные разности) ----------
        for (size_t i = 0; i < n; ++i) {
            //printf("grad: %d of %d\n",i,n);
            double xi = x[i];
            double h = dx_abs + fabs(xi) * dx_rel; // адаптивный шаг
            x_shift = x;
            x_shift[i] = xi + h;
            double f_plus = f(x_shift, user_data);
            x_shift[i] = xi - h;
            double f_minus = f(x_shift, user_data);
            if (std::isnan(f_plus) || std::isinf(f_plus) ||
                std::isnan(f_minus) || std::isinf(f_minus)) {
                x_out = x;
                return false;
            }
            grad[i] = (f_plus - f_minus) / (2.0 * h);
        }
        //std::string fname = "grad.txt";
        //dump_v(fname,grad);
        //printf("iter %d: grad ok\n",iter);

        // ---------- 2. Линейный поиск (Armijo) ----------
        double grad_norm_sq = 0.0;
        for (size_t i = 0; i < n; ++i) grad_norm_sq += grad[i] * grad[i];
        double grad_norm = std::sqrt(grad_norm_sq);

        // Критерий остановки по градиенту
        if (grad_norm < epsilon) {
            converged = true;
            break;
        }

        // Направление спуска: антиградиент
        std::vector<double> dir(n);
        for (size_t i = 0; i < n; ++i) dir[i] = -grad[i];

        // Armijo: уменьшаем lambda, пока не выполнится f(x+lambda*dir) <= f(x) + c*lambda*grad^T*dir
        double f_next;
        bool armijo_ok = false;
        double lambda_try = lambda;

        // Защита от бесконечного цикла: не более 50 попыток
        for (int armijo_iter = 0; armijo_iter < 100; ++armijo_iter) {
            for (size_t i = 0; i < n; ++i)
                x_new[i] = x[i] + lambda_try * dir[i];
            f_next = f(x_new, user_data);
            if (std::isnan(f_next) || std::isinf(f_next)) {
                lambda_try *= lambda_reduce;
                continue;
            }
            double expected_reduction = -c_armijo * lambda_try * (grad_norm_sq);
            if (f_next <= f_cur + expected_reduction) {
                armijo_ok = true;
                // lambda = lambda_try;
                //printf("currently: %d\n",armijo_iter);
                break;
            }
            lambda_try *= lambda_reduce;
            if(lambda_try < 1e-15)
            {
                break;
            }
        }

        if (!armijo_ok) {
            // Линейный поиск не удался – возможно, уже в минимуме или функция негладкая
            // Всё равно обновляем с текущим lambda
            //printf("iter %d: armijo not ok\n",iter);
            lambda_try = 1e-15;
            for (size_t i = 0; i < n; ++i)
                x_new[i] = x[i] + lambda_try * dir[i];
            f_next = f(x_new, user_data);
            if (std::isnan(f_next) || std::isinf(f_next)) {
                x_out = x;
                return false;
            }
        }
        /*else
        {
            printf("iter %d: armijo ok\n",iter);
        }*/

        // ---------- 3. Обновление переменных и вычисление изменений ----------
        double x_norm_change = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double diff = x_new[i] - x[i];
            x_norm_change += diff * diff;
        }
        x_norm_change = std::sqrt(x_norm_change);

        double f_change = fabs(f_next - f_cur);
        double f_plus = fabs(f_next+f_cur);
        double f_rel_change = f_change / f_plus;
        if(std::isnan(f_rel_change)||std::isinf(f_rel_change))
        {
            f_rel_change = 0;
        }

        // Критерии остановки
        if (x_norm_change < epsilon && f_rel_change < epsilon) {
            converged = true;
        }

        // Принимаем шаг
        x = x_new;
        f_cur = f_next;

        // Адаптация lambda для следующей итерации (увеличиваем, если шаг был удачным)
        if (armijo_ok && lambda_try == lambda) {
            lambda = std::min(lambda * lambda_increase, 1.0);
        } else {
            lambda = lambda_try; // сохраняем подобранное значение
        }

        // Предотвращаем слишком маленький lambda
        if (lambda < 1e-12) lambda = 1e-12;
    }
    if(converged)
    {
        printf("Converged after %d iters\n",iter);
    }
    x_out = x;
    return converged;
}

/**
 * @brief linear_regression_basis
 * Производит линейную регрессию с базисом.
 * @param f_basis
 * Возвращает вектор базисных функций для каждого x_i
 * @param x_n
 * Вектор входных данных
 * @param w_out
 * Выход: веса
 * @param user_data
 * Пользовательские данные
 * @return
 */
bool linear_regression_basis(std::function<std::vector<double>(std::vector<double>&,void *)>f_basis,std::vector<double> &y,std::vector<std::vector<double>>&x_n,std::vector<double>&w_out,void *user_data)
{
    std::vector<double> F,Ft,FFt,Fp;
    std::vector<double> fn;
    assert(x_n.size()>0);
    assert(x_n[0].size()>0);
    fn = f_basis(x_n[0],user_data);
    size_t N_basis = fn.size();
    size_t N_data = x_n.size();
    F.resize(N_basis * N_data);
    Ft.resize(N_basis * N_data);
    FFt.resize(N_basis * N_basis);
    for(size_t i = 0;i<N_data;i++)
    {
        fn = f_basis(x_n[i],user_data);
        for(size_t j=0;j<fn.size();j++)
        {
            F[i*N_basis+j] = fn[j];
            Ft[j*N_data+i] = fn[j];
        }
    }
    //Заполнили данные, умножаем.
    //Ft: [N_basis,N_data]
    //F: [N_data,N_basis]
    //Ft*F: [N_basis,N_basis]
    printf("matrix done\n");
    MatrixMatrixMultiply(FFt.data(),Ft.data(),F.data(),N_basis,N_data,N_basis);
    inverse(FFt.data(),N_basis);
    printf("inverse done\n");
    Fp.resize(N_basis*N_data);
    MatrixMatrixMultiply(Fp.data(),FFt.data(),Ft.data(),N_basis,N_basis,N_data);
    //Fp: [N_basis,N_data];
    w_out.resize(N_basis);
    MatrixMatrixMultiply(w_out.data(),Fp.data(),y.data(),N_basis,N_data,1);
    printf("all done\n");
    return true;
}

// Вспомогательная функция: численный градиент (центральные разности)
static void compute_gradient(std::function<double(std::vector<double>&, void*)>& f,
                             const std::vector<double>& x,
                             std::vector<double>& grad,
                             void* user_data)
{
    const double dx_rel = 1e-7;
    const double dx_abs = 1e-8;
    size_t n = x.size();
    grad.resize(n);
    std::vector<double> x_plus = x;
    std::vector<double> x_minus = x;
    for (size_t i = 0; i < n; ++i) {
        double h = dx_abs + std::fabs(x[i]) * dx_rel;
        x_plus[i] = x[i] + h;
        x_minus[i] = x[i] - h;
        double f_plus = f(x_plus, user_data);
        double f_minus = f(x_minus, user_data);
        grad[i] = (f_plus - f_minus) / (2.0 * h);
        // восстановление
        x_plus[i] = x[i];
        x_minus[i] = x[i];
    }
}

// Линейный поиск Armijo с бэктрекингом
// Возвращает alpha (длину шага) и новое значение функции f_new
static double armijo_backtracking(std::function<double(std::vector<double>&, void*)>& f,
                                  const std::vector<double>& x,
                                  const std::vector<double>& dir,
                                  double f_cur,
                                  const std::vector<double>& grad,
                                  void* user_data,
                                  double alpha_init = 1.0,
                                  double c = 1e-4,
                                  double rho = 0.5)
{
    double alpha = alpha_init;
    size_t n = x.size();
    std::vector<double> x_new(n);
    double grad_dir = 0.0;
    for (size_t i = 0; i < n; ++i) grad_dir += grad[i] * dir[i]; // grad^T * dir (отрицательно)

    while (alpha > 1e-12) {
        for (size_t i = 0; i < n; ++i) x_new[i] = x[i] + alpha * dir[i];
        double f_new = f(x_new, user_data);
        if (std::isnan(f_new) || std::isinf(f_new)) {
            alpha *= rho;
            continue;
        }
        if (f_new <= f_cur + c * alpha * grad_dir) {
            return alpha; // успех
        }
        alpha *= rho;
    }
    return 0.0; // не удалось
}

// Двухпетлевой L-BFGS для вычисления направления d = -H * g
static void lbfgs_direction(const std::vector<double>& grad,
                            std::vector<std::vector<double>>& s_list,
                            std::vector<std::vector<double>>& y_list,
                            std::vector<double>& rho_list,
                            std::vector<double>& d,
                            int m, int k)
{
    // k – текущее количество сохранённых пар (не более m)
    // Реализуем двухпетлевой алгоритм:
    // q = grad
    // for i = k-1 downto 0:
    //   alpha_i = rho_i * s_i^T * q
    //   q = q - alpha_i * y_i
    // r = H0 * q   (H0 = gamma * I, gamma = (s_{k-1}^T y_{k-1})/(y_{k-1}^T y_{k-1}) или 1)
    // for i = 0 to k-1:
    //   beta = rho_i * y_i^T * r
    //   r = r + s_i * (alpha_i - beta)
    // d = -r

    size_t n = grad.size();
    d.assign(n, 0.0);

    if (k == 0) {
        // Без истории – спуск по антиградиенту
        for (size_t i = 0; i < n; ++i) d[i] = -grad[i];
        return;
    }

    // Вычисляем gamma = s_{k-1}^T y_{k-1} / (y_{k-1}^T y_{k-1})
    double gamma = 1.0;
    if (k > 0) {
        double sdoty = 0.0, ydoty = 0.0;
        for (size_t i = 0; i < n; ++i) {
            sdoty += s_list[k-1][i] * y_list[k-1][i];
            ydoty += y_list[k-1][i] * y_list[k-1][i];
        }
        if (ydoty > 1e-12) gamma = sdoty / ydoty;
    }

    std::vector<double> alpha(k, 0.0);
    std::vector<double> q = grad;  // копия

    // Первый проход (обратный)
    for (int i = k-1; i >= 0; --i) {
        alpha[i] = rho_list[i];
        double dot = 0.0;
        for (size_t j = 0; j < n; ++j) dot += s_list[i][j] * q[j];
        alpha[i] *= dot;
        for (size_t j = 0; j < n; ++j) q[j] -= alpha[i] * y_list[i][j];
    }

    // Начальное приближение r = gamma * q
    std::vector<double> r(n);
    for (size_t i = 0; i < n; ++i) r[i] = gamma * q[i];

    // Второй проход (прямой)
    for (int i = 0; i < k; ++i) {
        double beta = rho_list[i];
        double dot = 0.0;
        for (size_t j = 0; j < n; ++j) dot += y_list[i][j] * r[j];
        beta *= dot;
        for (size_t j = 0; j < n; ++j) r[j] += s_list[i][j] * (alpha[i] - beta);
    }

    // Направление: отрицательное
    for (size_t i = 0; i < n; ++i) d[i] = -r[i];
}

bool find_minimum_lbfgs(std::function<double(std::vector<double> &x,void *user_data)> f, const std::vector<double> &x_init, std::vector<double> &x_out, void *user_data, double lambda_init, double epsilon, int max_iter, int m)
{
    size_t n = x_init.size();
    std::vector<double> x = x_init;
    std::vector<double> grad(n);
    std::vector<double> x_new(n);
    std::vector<double> x_shift(n);
    std::vector<std::vector<double>> H_k;
    double lambda = lambda_init;
    int k = 0;
    bool converged = false;
    //На начальном этапе задаем матрицу H0;

    while (k < max_iter && !converged) {
        //Шаг 1: расчет градиента.
        compute_gradient(f,x,grad,user_data);

    }
}


bool is_sym(matrix_t m)
{
    if(m.size()==0 || m[0].size()!=m.size()){
        return false;
    }
    for(size_t i=0;i<m.size();i++){
        for(size_t j=0;j<m[0].size();j++){
            if(m[i][j]!=m[j][i]){
                return false;
            }
        }
    }
    return true;
}
bool is_herm(cmatrix_t m)
{
    if(m.size()==0 || m[0].size()!=m.size()){
        return false;
    }
    for(size_t i=0;i<m.size();i++){
        for(size_t j=0;j<m[0].size();j++){
            if((creal(m[i][j])!=creal(m[j][i]))||(cimag(m[i][j])!=-cimag(m[j][i]))){
                return false;
            }
        }
    }
    return true;
}

double det(int N, double *a)
{
    int lda,*ipiv,info;
    double rv;
    lda = N;
    ipiv = new int[N];
    dgetrf_(&N,&N,a,&lda,ipiv,&info);
    rv = 1.0;
    for(int i=0;i<N;i++)
    {
        rv *= a[i*N+i];
    }
    for(int i=0;i<N;i++)
    {
        if(ipiv[i]!=(i+1))
        {
            rv=-rv;
        }
    }
    return rv;
}

double_complex_t zdet(int N, double_complex_t *m) {
    double_complex_t **a,*dummy;
    a=(double_complex_t **)malloc(N*sizeof(double_complex_t *));
    dummy = (double_complex_t *)malloc(N*sizeof(double_complex_t));
    for(int i=0;i<N;i++)
    {
        a[i]=(double_complex_t *)malloc(N*sizeof(double_complex_t));
    }
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
        {
            a[i][j]=m[i*N+j];
        }
    }
    double_complex_t det = 1.0;
    for (int i = 0; i < N; i++) {
        int pivot = i;
        for (int j = i + 1; j < N; j++) {
            if (cabs(a[j][i]) > cabs(a[pivot][i])) {
                pivot = j;
            }
        }
        if (pivot != i) {
            for(int k=0;k<N;k++)
            {
                dummy[k]=a[i][k];
                a[i][k]=a[pivot][k];
                a[pivot][k]=dummy[k];
            }
            det *= -1;
        }
        if (a[i][i] == 0) {
            free(dummy);
            for(int i=0;i<N;i++)
            {
                free(a[i]);
            }
            free(a);
            return 0;
        }
        det *= a[i][i];
        for (int j = i + 1; j < N; j++) {
            double_complex_t factor = a[j][i] / a[i][i];
            for (int k = i + 1; k <N; k++) {
                a[j][k] -= factor * a[i][k];
            }
        }
    }
    free(dummy);
    for(int i=0;i<N;i++)
    {
        free(a[i]);
    }
    free(a);
    return det;
}
/*
int integrand(const int *ndim_intern, const double xx[],const int *ncomp, double ff[], void *userdata)
{
    cuba_t *p;
    double *t;
    double jacobian=1;
    int i;
    p=(cuba_t *)userdata;
    t=(double *)malloc(p->ndim*sizeof(double));//Translated to [a[i],b[i]]
    for(i=0;i<p->ndim;i++){
        t[i]=(xx[i])*(p->b[i]-p->a[i])+p->a[i];
        jacobian*=p->b[i]-p->a[i];
    }
    ff[0]=jacobian*p->f(t,p->sd);
}

int integrand_real(const int *ndim_intern, const double xx[],const int *ncomp, double ff[], void *userdata) {
    zcuba_t *p;
    double *t;
    double jacobian=1;
    int i;
    p=(zcuba_t *)userdata;
    t=(double *)malloc(p->ndim*sizeof(double));//Translated to [a[i],b[i]]
    for(i=0;i<p->ndim;i++){
        t[i]=(xx[i])*(p->b[i]-p->a[i])+p->a[i];
        jacobian*=p->b[i]-p->a[i];
    }
    ff[0]=jacobian*creal(p->f(t,p->sd));
}
int integrand_imag(const int *ndim_intern, const double xx[],const int *ncomp, double ff[], void *userdata) {
    zcuba_t *p;
    double *t;
    double jacobian=1;
    int i;
    p=(zcuba_t *)userdata;
    t=(double *)malloc(p->ndim*sizeof(double));//Translated to [a[i],b[i]]
    for(i=0;i<p->ndim;i++){
        t[i]=(xx[i])*(p->b[i]-p->a[i])+p->a[i];
        jacobian*=p->b[i]-p->a[i];
    }
    ff[0]=jacobian*cimag(p->f(t,p->sd));
}


double GaussIntegrateCuba(double (*f)(double[],void *),void *serviceData,int ndim,double a[],double b[],double epsrel,double epsabs){
    int nvec_glob=1;
    int comp, nregions, neval, fail;
    cuba_t p;
    p.f = f;
    p.sd = serviceData;
    p.ndim = ndim;
    p.a = a;
    p.b = b;
    double integral[NCOMP], error[NCOMP], prob[NCOMP];
    Cuhre(ndim, NCOMP, integrand, (void *)&p, nvec_glob,epsrel, epsabs, VERBOSE | LAST,MINEVAL,MAXEVAL, KEY,STATEFILE, SPIN,&nregions, &neval, &fail, integral, error, prob);
    return integral[0];
}

double_complex_t ZGaussIntegrateCuba(double_complex_t (*f)(double[],void *),void *serviceData,int ndim,double a[],double b[],double epsrel,double epsabs){
    int nvec_glob=1;
    int comp, nregions, neval, fail;
    double integral_r[NCOMP],integral_i[NCOMP], error[NCOMP], prob[NCOMP];
    zcuba_t p;
    p.f = f;
    p.sd = serviceData;
    p.ndim = ndim;
    p.a = a;
    p.b = b;

    Cuhre(ndim, NCOMP, integrand_real, (void *)&p, nvec_glob,epsrel, epsabs, VERBOSE | LAST,MINEVAL,MAXEVAL, KEY,STATEFILE, SPIN,&nregions, &neval, &fail, integral_r, error, prob);
    Cuhre(ndim, NCOMP, integrand_imag, (void *)&p, nvec_glob,epsrel, epsabs, VERBOSE | LAST,MINEVAL,MAXEVAL, KEY,STATEFILE, SPIN,&nregions, &neval, &fail, integral_i, error, prob);
    return integral_r[0]+I*integral_i[0];
}
*/
/*std::vector<complex_t> eigen3_find_eigenvals(std::vector<std::vector<complex_t>> &mat)
{
    Eigen::MatrixXcd system;
    Eigen::MatrixXcd res;
    std::vector<complex_t> out;
    size_t cols,rows;
    rows = mat.size();
    assert(rows>0);
    cols = mat[0].size();
    assert(cols == rows);
    system.resize(rows,cols);
    for(size_t i=0;i<rows;i++)
    {
        for(size_t j=0;j<cols;j++)
        {
            system(i,j) = mat[i][j];
        }
    }
    Eigen::ComplexEigenSolver<Eigen::MatrixXcd> solver;
    solver.compute(system);
    res = solver.eigenvalues();
    // std::cout << "Matrix A is:" << std::endl << system << std::endl;
    std::cout << "The eigenvalues of A are:" << std::endl << solver.eigenvalues() << std::endl;
    out.resize(rows);
    // printf("%d\n",res.size());
    // for(int i=0;i<rows;i++)
    // {
    // printf("i=%d\n",i);
    // out[i] = res(i);
    // }
    return out;
}*/

void EigenSystemComplex(std::vector<std::vector<complex_t>> &a,std::vector<complex_t> &w,std::vector<std::vector<complex_t>> &zc,bool eigenvectors)
{
    char job='V';
    char job_nothing='N';
    char up='U';
    lapack_complex_t *ww,*vr;
    double *rw;
    lapack_complex_t *m,*work;
    int info;
    int n;
    assert(a.size()>0);
    assert(a[0].size()==a.size());
    n = a.size();
    w.clear();
    w.resize(n);
    m=new lapack_complex_t[n*n];
    vr=new lapack_complex_t[n*n];
    ww=new lapack_complex_t[n];
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            m[i*n+j].real=a[i][j].real();
            m[i*n+j].imag=a[i][j].imag();
        }
    }
    int lwork=32*n;
    int rwork=32*n;
    work=new lapack_complex_t[lwork];
    rw=new double[rwork];
    zgeev_(&job_nothing,&job,&n,m,&n,ww,NULL,&n,vr,&n,work,&lwork,rw,&info);
    std::vector<complex_t> cur;
    for(int i=0;i<n;i++){
        w[i]=ww[i].real+ww[i].imag*_i;
    }
    zc.clear();
    if(eigenvectors){
        for(int j=0;j<n;j++){
            cur.clear();
            cur.resize(n);
            for(int i=0;i<n;i++){
                cur[i]=vr[j*n+i].real+_i*vr[j*n+i].imag;
            }
            zc.push_back(cur);
        }
    }
    delete []m;
    delete []work;
    delete []ww;
    delete []rw;
    delete []vr;
}

double to_2f1(double *x_,void *sd)
{
    double x,a,b,c,z;
    hypergeom2f1_params *p;
    p = (hypergeom2f1_params *)sd;
    x = x_[0];
    a = p->a;
    b = p->b;
    c = p->c;
    z = p->z;
    return pow(x,b-1)*pow(1.0-x,c-b-1)*pow(1-x*z,-a);
}

double Hypergeometric2F1(double a,double b,double c,double z)
{
    hypergeom2f1_params p;
    double left[1],right[1];
    double rv;
    p.a = a;
    p.b = b;
    p.c = c;
    p.z = z;
    left[0] = 0.0;
    right[0] = 1.0;
    rv = (Gamma(c)/(Gamma(b)*Gamma(c-b)))*GaussIntegrateParallel(to_2f1,(void *)&p,1,left,right,20,get_num_threads());
    return rv;
}
//TRASH FROM APPELLF4
/*
    double sum = 0;
    double eps = 1e-16;
    double term=1;
    int m,n,maxiter;
    maxiter = 10;
    for(m=0;m<maxiter;m++)
    {
        for(n=0;n<maxiter;n++)
        {
            term = ipow(x,m)*ipow(y,n)*Pochgammer(m+n,a)*Pochgammer(m+n,b)/(Pochgammer(m,c1)*Pochgammer(n,c2)*ifact(m)*ifact(n));
            sum+=term;
        }
    }
    return sum;
*/
