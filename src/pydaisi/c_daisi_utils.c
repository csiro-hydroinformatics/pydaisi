#include "c_daisi_utils.h"

/* See
 * https://stackoverflow.com/questions/101439/the-most-efficient-way-to-implement-an-integer-based-power-function-powint-int
 * */
int ipow(int base, int exp)
{
    int result = 1;
    for (;;)
    {
        if (exp & 1)
            result *= base;
        exp >>= 1;
        if (!exp)
            break;
        base *= base;
    }

    return result;
}

double get_nan() {
    double zero=0.;
    static double nan;
    nan = zero/zero;
    return nan;
}

double c_gr2m_S_fun(double X1, double S, double P, double E) {
    double PHI = c_tanh(P/X1);
    double S1 = (S+X1*PHI)/(1+PHI*S/X1);
    double PSI = c_tanh(E/X1);
    double S2 = S1*(1-PSI)/(1+PSI*(1-S1/X1));
    double Sr = S2/X1;
    return S2/cbrt(1.+Sr*Sr*Sr);
}

double c_gr2m_P3_fun(double X1, double S, double P, double E) {
    double PHI = c_tanh(P/X1);
    double S1 = (S+X1*PHI)/(1+PHI*S/X1);
    double P1 = P+S-S1;
    double PSI = c_tanh(E/X1);
    double S2 = S1*(1-PSI)/(1+PSI*(1-S1/X1));
    double Sr = S2/X1;
    double Send = S2/cbrt(1.+Sr*Sr*Sr);
    double P2 = S2-Send;
    return P1+P2;
}

double c_gr2m_R_fun(double X2, double Xr, double R, double P3) {
    double R1 = R + P3;
    double R2 = X2*R1;
    return R2*Xr/(R2+Xr);
}

double c_gr2m_Q_fun(double X2, double Xr, double R, double P3) {
    double R1 = R + P3;
    double R2 = X2*R1;
    return R2*R2/(R2+Xr);
}

/* Normalisation functions */
/* .. production .. */
double c_gr2m_prod_S_raw2norm(double X1, double S){
    return S/X1;
}
double c_gr2m_prod_S_norm2raw(double X1, double u){
    return u*X1;
}

double c_gr2m_prod_P_raw2norm(double X1, double P){
    return P/X1;
}
double c_gr2m_prod_P_norm2raw(double X1, double pn){
    return X1*pn;
}

double c_gr2m_prod_E_raw2norm(double X1, double E){
    return E/X1;
}
double c_gr2m_prod_E_norm2raw(double X1, double en){
    return X1*en;
}

double c_gr2m_prod_AE_raw2norm(double X1, double AE){
    return AE/X1;
}
double c_gr2m_prod_AE_norm2raw(double X1, double aen){
    return X1*aen;
}

double c_gr2m_prod_P3_raw2norm(double X1, double P3){
    return P3/X1;
}
double c_gr2m_prod_P3_norm2raw(double X1, double p3n){
    return X1*p3n;
}

/* .. routing .. */
double c_gr2m_rout_P3_raw2norm(double X2, double Xr, double P3){
    return X2/Xr*P3;
}
double c_gr2m_rout_P3_norm2raw(double X2, double Xr, double p3n){
    return Xr/X2*p3n;
}

double c_gr2m_rout_Rstart_raw2norm(double X2, double Xr, double R){
    return X2/Xr*R;
}
double c_gr2m_rout_Rstart_norm2raw(double X2, double Xr, double v){
    return Xr/X2*v;
}

double c_gr2m_rout_Rend_raw2norm(double X2, double Xr, double R){
    return R/Xr;
}
double c_gr2m_rout_Rend_norm2raw(double X2, double Xr, double v){
    return v*Xr;
}

double c_gr2m_rout_F_raw2norm(double X2, double Xr, double F){
    return X2/Xr/(X2-1)*F;
}
double c_gr2m_rout_F_norm2raw(double X2, double Xr, double fn){
    return (X2-1)*Xr/X2*fn;
}

double c_gr2m_rout_Q_raw2norm(double X2, double Xr, double Q){
    return Q/Xr;
}
double c_gr2m_rout_Q_norm2raw(double X2, double Xr, double qn){
    return Xr*qn;
}


/* Box cox transform functions */

double c_boxcox_forward(double x, double lam, double nu) {
    double u=x+nu;

    if(u<=0)
        return get_nan();

    /* Bypass power function if lam= 0, -1, -0.5, 0.5, 1*/
    if(fabs(lam)<1e-10)
        return log(u);

    else if(fabs(lam+1)<1e-10)
        return 1.-1./u;

    else if(fabs(lam+0.5)<1e-10)
        return 2.*(1.-1./sqrt(u));

    else if(fabs(lam-0.5)<1e-10)
        return 2*(sqrt(u)-1);

    else if(fabs(lam-1.)<1e-10)
        /* 0 bounds not necessary here, added for continuity with other cases
         * */
        return u-1;

    return (pow(u, lam)-1.)/lam;
}


double c_boxcox_backward(double y, double lam, double nu) {
    double v = lam*y+1.;

    if(v<=0)
        return get_nan();

    /* Bypass power function if lam= 0, -1, -0.5, 0.5, 1*/
    if(fabs(lam)<1e-10)
        return exp(y)-nu;

    /* .. 0 bounds not necessary here, added for continuity with
     *     the general case
     */
    else if(fabs(lam+1)<1e-10)
        return 1./v-nu;

    else if(fabs(lam+0.5)<1e-10)
        return (1./v)*(1./v)-nu;

    else if(fabs(lam-0.5)<1e-10)
        return v*v-nu;

    else if(fabs(lam-1.)<1e-10)
        return v-nu;

    return pow(v, 1./lam)-nu;
}


double c_boxcox_perturb(double x, double delta, double lam, double nu, \
                                double xclip) {
    double y = c_boxcox_forward(x, lam, nu)+delta;
    double ymax;
    double yclip = c_boxcox_forward(xclip, lam, nu);
    double xp = c_boxcox_backward(y, lam, nu);

    /* Clipping threshold is NaN -> No clipping */
    if(isnan(xclip))
        return xp;

    /* if delta = 0, returns max(xclip, x) */
    if(fabs(delta)<1e-10)
        return x>xclip ? x : xclip;

    /* Case where lam<0 and y is above the -1/lam
    * Need to clip up
    */
    if(lam<0) {
        ymax = -1/lam-1e-2;
        xp = y>ymax ? c_boxcox_backward(ymax, lam, nu): xp;
    }

    /* Does clipping if clipping threshold is valid (i.e. yclip not nan) */
    if(isnan(yclip))
        return get_nan();
    else
        return y>=yclip ? xp : xclip;
}



