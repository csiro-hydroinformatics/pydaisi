#ifndef __DUTILS__
#define __DUTILS__

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "c_utils.h"

#define C_PI 3.1415926535897932384626433832795

/* Define Error message */
#define DAISI_ERROR 80000
#define GR2M_UPDATE_ERROR 90000

/* Maximum for the argument of atanh function (i.e. atanh(x)<18) */
#define ATANH_MAX 0.999999999999999

/* Maximum for the argument of x/(1-x) function (i.e. x/(1-x)<100) */
#define RATIONAL_MAX 0.99


double c_gr2m_S_fun(double X1, double S, double P, double E);
double c_gr2m_P3_fun(double X1, double S, double P, double E);
double c_gr2m_R_fun(double X2, double Xr, double R, double P3);
double c_gr2m_Q_fun(double X2, double Xr, double R, double P3);

double c_gr2m_prod_S_raw2norm(double X1, double S);
double c_gr2m_prod_S_norm2raw(double X1, double u);
double c_gr2m_prod_P_raw2norm(double X1, double P);
double c_gr2m_prod_P_norm2raw(double X1, double pn);
double c_gr2m_prod_E_raw2norm(double X1, double E);
double c_gr2m_prod_E_norm2raw(double X1, double en);
double c_gr2m_prod_AE_raw2norm(double X1, double AE);
double c_gr2m_prod_AE_norm2raw(double X1, double aen);
double c_gr2m_prod_P3_raw2norm(double X1, double P3);
double c_gr2m_prod_P3_norm2raw(double X1, double p3n);

double c_gr2m_rout_P3_raw2norm(double X2, double Xr, double P3);
double c_gr2m_rout_P3_norm2raw(double X2, double Xr, double p3n);
double c_gr2m_rout_Rstart_raw2norm(double X2, double Xr, double R);
double c_gr2m_rout_Rstart_norm2raw(double X2, double Xr, double v);
double c_gr2m_rout_Rend_raw2norm(double X2, double Xr, double R);
double c_gr2m_rout_Rend_norm2raw(double X2, double Xr, double v);
double c_gr2m_rout_F_raw2norm(double X2, double Xr, double F);
double c_gr2m_rout_F_norm2raw(double X2, double Xr, double fn);
double c_gr2m_rout_Q_raw2norm(double X2, double Xr, double Q);
double c_gr2m_rout_Q_norm2raw(double X2, double Xr, double qn);


double c_boxcox_forward(double x, double lam, double nu);
double c_boxcox_backward(double y, double lam, double nu);
double c_boxcox_perturb(double x, double delta, double lam, double nu, double xclip);

#endif

