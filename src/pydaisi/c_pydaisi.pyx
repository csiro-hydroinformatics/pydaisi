import numpy as np
cimport numpy as np

np.import_array()

# -- HEADERS --
cdef extern from "c_gr2m_update.h":
    int c_get_modif_params_start()

    int c_gr2m_update_run(int nval, int nconfig, int nparams,
            int ninputs,
            int nstates, int noutputs,
            int start, int end,
            double * config,
            double * params,
            double * inputs,
            double * statesini,
            double * outputs)

cdef extern from "c_daisi_utils.h":
    double c_boxcox_forward(double x, double lam, double nu)
    double c_boxcox_backward(double y, double lam, double nu)
    double c_boxcox_perturb(double x, double delta, double lam, double nu,
                                    double xclip)

    double c_gr2m_S_fun(double X1, double S, double P, double E)
    double c_gr2m_P3_fun(double X1, double S, double P, double E)
    double c_gr2m_R_fun(double X2, double Xr, double R, double P3)
    double c_gr2m_Q_fun(double X2, double Xr, double R, double P3)

    double c_gr2m_prod_S_raw2norm(double X1, double S)
    double c_gr2m_prod_S_norm2raw(double X1, double u)
    double c_gr2m_prod_P_raw2norm(double X1, double P)
    double c_gr2m_prod_P_norm2raw(double X1, double pn)
    double c_gr2m_prod_E_raw2norm(double X1, double E)
    double c_gr2m_prod_E_norm2raw(double X1, double en)
    double c_gr2m_prod_AE_raw2norm(double X1, double AE)
    double c_gr2m_prod_AE_norm2raw(double X1, double aen)
    double c_gr2m_prod_P3_raw2norm(double X1, double P3)
    double c_gr2m_prod_P3_norm2raw(double X1, double p3n)

    double c_gr2m_rout_P3_raw2norm(double X2, double Xr, double P3)
    double c_gr2m_rout_P3_norm2raw(double X2, double Xr, double p3n)
    double c_gr2m_rout_Rstart_raw2norm(double X2, double Xr, double R)
    double c_gr2m_rout_Rstart_norm2raw(double X2, double Xr, double v)
    double c_gr2m_rout_Rend_raw2norm(double X2, double Xr, double R)
    double c_gr2m_rout_Rend_norm2raw(double X2, double Xr, double v)
    double c_gr2m_rout_F_raw2norm(double X2, double Xr, double F)
    double c_gr2m_rout_F_norm2raw(double X2, double Xr, double fn)
    double c_gr2m_rout_Q_raw2norm(double X2, double Xr, double Q)
    double c_gr2m_rout_Q_norm2raw(double X2, double Xr, double qn)

def __cinit__(self):
    pass


def get_modif_params_start():
    return c_get_modif_params_start()


# GR2M functions
def gr2m_S_fun(double X1, double S, double P, double E):
    return c_gr2m_S_fun(X1, S, P, E)

def vect_gr2m_S_fun(double X1,
        np.ndarray[double, ndim=1, mode="c"] S not None,
        np.ndarray[double, ndim=1, mode="c"] P not None,
        np.ndarray[double, ndim=1, mode="c"] E not None,
        np.ndarray[double, ndim=1, mode="c"] out not None):
    cdef int i
    cdef int nval = S.shape[0]
    assert P.shape[0] == nval
    assert E.shape[0] == nval
    assert out.shape[0] == nval
    for i in range(nval):
        out[i] =  c_gr2m_S_fun(X1, S[i], P[i], E[i])

def gr2m_P3_fun(double X1, double S, double P, double E):
    return c_gr2m_P3_fun(X1, S, P, E)

def vect_gr2m_P3_fun(double X1,
        np.ndarray[double, ndim=1, mode="c"] S not None,
        np.ndarray[double, ndim=1, mode="c"] P not None,
        np.ndarray[double, ndim=1, mode="c"] E not None,
        np.ndarray[double, ndim=1, mode="c"] out not None):
    cdef int i
    cdef int nval = S.shape[0]
    assert P.shape[0] == nval
    assert E.shape[0] == nval
    assert out.shape[0] == nval
    for i in range(nval):
        out[i] =  c_gr2m_P3_fun(X1, S[i], P[i], E[i])

def gr2m_R_fun(double X2, double Xr, double R, double P3):
    return c_gr2m_R_fun(X2, Xr, R, P3)

def vect_gr2m_R_fun(double X2, double Xr,
        np.ndarray[double, ndim=1, mode="c"] R not None,
        np.ndarray[double, ndim=1, mode="c"] P3 not None,
        np.ndarray[double, ndim=1, mode="c"] out not None):
    cdef int i
    cdef int nval = R.shape[0]
    assert P3.shape[0] == nval
    assert out.shape[0] == nval
    for i in range(nval):
        out[i] =  c_gr2m_R_fun(X2, Xr, R[i], P3[i])

def gr2m_Q_fun(double X2, double Xr, double R, double P3):
    return c_gr2m_Q_fun(X2, Xr, R, P3)

def vect_gr2m_Q_fun(double X2, double Xr,
        np.ndarray[double, ndim=1, mode="c"] R not None,
        np.ndarray[double, ndim=1, mode="c"] P3 not None,
        np.ndarray[double, ndim=1, mode="c"] out not None):
    cdef int i
    cdef int nval = R.shape[0]
    assert P3.shape[0] == nval
    assert out.shape[0] == nval
    for i in range(nval):
        out[i] =  c_gr2m_Q_fun(X2, Xr, R[i], P3[i])


# Normalisation function
def vect_gr2m_prod_S_raw2norm(double X1,
        np.ndarray[double, ndim=1, mode="c"] S not None,
        np.ndarray[double, ndim=1, mode="c"] out not None):
    cdef int i
    cdef int nval = S.shape[0]
    assert out.shape[0] == nval
    for i in range(nval):
        out[i] = c_gr2m_prod_S_raw2norm(X1, S[i])

def vect_gr2m_prod_S_norm2raw(double X1,
        np.ndarray[double, ndim=1, mode="c"] u not None,
        np.ndarray[double, ndim=1, mode="c"] out not None):
    cdef int i
    cdef int nval = u.shape[0]
    assert out.shape[0] == nval
    for i in range(nval):
        out[i] = c_gr2m_prod_S_norm2raw(X1, u[i])

def vect_gr2m_prod_P_raw2norm(double X1,
        np.ndarray[double, ndim=1, mode="c"] P not None,
        np.ndarray[double, ndim=1, mode="c"] out not None):
    cdef int i
    cdef int nval = P.shape[0]
    assert out.shape[0] == nval
    for i in range(nval):
        out[i] = c_gr2m_prod_P_raw2norm(X1, P[i])

def vect_gr2m_prod_P_norm2raw(double X1,
        np.ndarray[double, ndim=1, mode="c"] pn not None,
        np.ndarray[double, ndim=1, mode="c"] out not None):
    cdef int i
    cdef int nval = pn.shape[0]
    assert out.shape[0] == nval
    for i in range(nval):
        out[i] = c_gr2m_prod_P_norm2raw(X1, pn[i])

def vect_gr2m_prod_E_raw2norm(double X1,
        np.ndarray[double, ndim=1, mode="c"] E not None,
        np.ndarray[double, ndim=1, mode="c"] out not None):
    cdef int i
    cdef int nval = E.shape[0]
    assert out.shape[0] == nval
    for i in range(nval):
        out[i] = c_gr2m_prod_E_raw2norm(X1, E[i])

def vect_gr2m_prod_E_norm2raw(double X1,
        np.ndarray[double, ndim=1, mode="c"] en not None,
        np.ndarray[double, ndim=1, mode="c"] out not None):
    cdef int i
    cdef int nval = en.shape[0]
    assert out.shape[0] == nval
    for i in range(nval):
        out[i] = c_gr2m_prod_E_norm2raw(X1, en[i])

def vect_gr2m_prod_AE_raw2norm(double X1,
        np.ndarray[double, ndim=1, mode="c"] AE not None,
        np.ndarray[double, ndim=1, mode="c"] out not None):
    cdef int i
    cdef int nval = AE.shape[0]
    assert out.shape[0] == nval
    for i in range(nval):
        out[i] = c_gr2m_prod_AE_raw2norm(X1, AE[i])

def vect_gr2m_prod_AE_norm2raw(double X1,
        np.ndarray[double, ndim=1, mode="c"] aen not None,
        np.ndarray[double, ndim=1, mode="c"] out not None):
    cdef int i
    cdef int nval = aen.shape[0]
    assert out.shape[0] == nval
    for i in range(nval):
        out[i] = c_gr2m_prod_AE_norm2raw(X1, aen[i])


def vect_gr2m_prod_P3_raw2norm(double X1,
        np.ndarray[double, ndim=1, mode="c"] P3 not None,
        np.ndarray[double, ndim=1, mode="c"] out not None):
    cdef int i
    cdef int nval = P3.shape[0]
    assert out.shape[0] == nval
    for i in range(nval):
        out[i] = c_gr2m_prod_P3_raw2norm(X1, P3[i])

def vect_gr2m_prod_P3_norm2raw(double X1,
        np.ndarray[double, ndim=1, mode="c"] p3n not None,
        np.ndarray[double, ndim=1, mode="c"] out not None):
    cdef int i
    cdef int nval = p3n.shape[0]
    assert out.shape[0] == nval
    for i in range(nval):
        out[i] = c_gr2m_prod_P3_norm2raw(X1, p3n[i])

def vect_gr2m_rout_P3_raw2norm(double X2, double Xr,
        np.ndarray[double, ndim=1, mode="c"] P3 not None,
        np.ndarray[double, ndim=1, mode="c"] out not None):
    cdef int i
    cdef int nval = P3.shape[0]
    assert out.shape[0] == nval
    for i in range(nval):
        out[i] = c_gr2m_rout_P3_raw2norm(X2, Xr, P3[i])

def vect_gr2m_rout_P3_norm2raw(double X2, double Xr,
        np.ndarray[double, ndim=1, mode="c"] p3n not None,
        np.ndarray[double, ndim=1, mode="c"] out not None):
    cdef int i
    cdef int nval = p3n.shape[0]
    assert out.shape[0] == nval
    for i in range(nval):
        out[i] = c_gr2m_rout_P3_norm2raw(X2, Xr, p3n[i])

def vect_gr2m_rout_Rstart_raw2norm(double X2, double Xr,
        np.ndarray[double, ndim=1, mode="c"] R not None,
        np.ndarray[double, ndim=1, mode="c"] out not None):
    cdef int i
    cdef int nval = R.shape[0]
    assert out.shape[0] == nval
    for i in range(nval):
        out[i] = c_gr2m_rout_Rstart_raw2norm(X2, Xr, R[i])

def vect_gr2m_rout_Rstart_norm2raw(double X2, double Xr,
        np.ndarray[double, ndim=1, mode="c"] v not None,
        np.ndarray[double, ndim=1, mode="c"] out not None):
    cdef int i
    cdef int nval = v.shape[0]
    assert out.shape[0] == nval
    for i in range(nval):
        out[i] = c_gr2m_rout_Rstart_norm2raw(X2, Xr, v[i])

def vect_gr2m_rout_Rend_raw2norm(double X2, double Xr,
        np.ndarray[double, ndim=1, mode="c"] R not None,
        np.ndarray[double, ndim=1, mode="c"] out not None):
    cdef int i
    cdef int nval = R.shape[0]
    assert out.shape[0] == nval
    for i in range(nval):
        out[i] = c_gr2m_rout_Rend_raw2norm(X2, Xr, R[i])

def vect_gr2m_rout_Rend_norm2raw(double X2, double Xr,
        np.ndarray[double, ndim=1, mode="c"] v not None,
        np.ndarray[double, ndim=1, mode="c"] out not None):
    cdef int i
    cdef int nval = v.shape[0]
    assert out.shape[0] == nval
    for i in range(nval):
        out[i] = c_gr2m_rout_Rend_norm2raw(X2, Xr, v[i])


def vect_gr2m_rout_F_raw2norm(double X2, double Xr,
        np.ndarray[double, ndim=1, mode="c"] F not None,
        np.ndarray[double, ndim=1, mode="c"] out not None):
    cdef int i
    cdef int nval = F.shape[0]
    assert out.shape[0] == nval
    for i in range(nval):
        out[i] = c_gr2m_rout_F_raw2norm(X2, Xr, F[i])

def vect_gr2m_rout_F_norm2raw(double X2, double Xr,
        np.ndarray[double, ndim=1, mode="c"] fn not None,
        np.ndarray[double, ndim=1, mode="c"] out not None):
    cdef int i
    cdef int nval = fn.shape[0]
    assert out.shape[0] == nval
    for i in range(nval):
        out[i] = c_gr2m_rout_F_norm2raw(X2, Xr, fn[i])


def vect_gr2m_rout_Q_raw2norm(double X2, double Xr,
        np.ndarray[double, ndim=1, mode="c"] Q not None,
        np.ndarray[double, ndim=1, mode="c"] out not None):
    cdef int i
    cdef int nval = Q.shape[0]
    assert out.shape[0] == nval
    for i in range(nval):
        out[i] = c_gr2m_rout_Q_raw2norm(X2, Xr, Q[i])

def vect_gr2m_rout_Q_norm2raw(double X2, double Xr,
        np.ndarray[double, ndim=1, mode="c"] qn not None,
        np.ndarray[double, ndim=1, mode="c"] out not None):
    cdef int i
    cdef int nval = qn.shape[0]
    assert out.shape[0] == nval
    for i in range(nval):
        out[i] = c_gr2m_rout_Q_norm2raw(X2, Xr, qn[i])


# Box cox functions
def boxcox_forward(double x, double lam, double nu):
    return c_boxcox_forward(x, lam, nu)

def boxcox_backward(double y, double lam, double nu):
    return c_boxcox_backward(y, lam, nu)

def boxcox_perturb(double x, double delta, double lam, double nu, double xclip):
    return c_boxcox_perturb(x, delta, lam, nu, xclip)


def boxcox_forward_vect(double lam, double nu,
                    np.ndarray[double, ndim=2, mode="c"] x not None,
                    np.ndarray[double, ndim=2, mode="c"] y not None):

    cdef int nval=x.shape[0]
    cdef int nvar=x.shape[1]
    assert y.shape[0] == nval, "x.shape[0]!=y.shape[0]"
    assert y.shape[1] == nvar, "x.shape[1]!=y.shape[1]"
    for i in range(nval):
        for j in range(nvar):
            y[i, j] = c_boxcox_forward(x[i, j], lam, nu)


def boxcox_backward_vect(double lam, double nu,
                    np.ndarray[double, ndim=2, mode="c"] y not None,
                    np.ndarray[double, ndim=2, mode="c"] x not None):

    cdef int nval=x.shape[0]
    cdef int nvar=x.shape[1]
    assert y.shape[0] == nval, "x.shape[0]!=y.shape[0]"
    assert y.shape[1] == nvar, "x.shape[1]!=y.shape[1]"
    for i in range(nval):
        for j in range(nvar):
            x[i, j] = c_boxcox_backward(y[i, j], lam, nu)


def boxcox_perturb_vect(double lam, double nu, double xclip,
                    np.ndarray[double, ndim=2, mode="c"] x not None,
                    np.ndarray[double, ndim=2, mode="c"] delta not None,
                    np.ndarray[double, ndim=2, mode="c"] y not None):
    cdef int i
    cdef int j
    cdef int nval = x.shape[0]
    cdef int nvarx = x.shape[1]
    cdef int nvar = delta.shape[1]
    cdef double xx
    assert y.shape[0] == nval, "x.shape[0]!=y.shape[0]"
    assert delta.shape[0] == nval, "x.shape[0]!=delta.shape[0]"
    assert y.shape[1] == nvar, "delta.shape[1]!=y.shape[1]"
    assert nvarx==nvar or nvarx==1, "x.shape[1]!=1 and x.shape[1]!=delta.shape[1]"

    for i in range(nval):
        for j in range(nvar):
            xx = x[i, 0] if nvarx==1 else x[i, j]
            y[i, j] = c_boxcox_perturb(xx, delta[i, j], lam, nu, xclip)



def gr2m_update_run(int start, int end,
        np.ndarray[double, ndim=1, mode="c"] config not None,
        np.ndarray[double, ndim=1, mode="c"] params not None,
        np.ndarray[double, ndim=2, mode="c"] inputs not None,
        np.ndarray[double, ndim=1, mode="c"] statesini not None,
        np.ndarray[double, ndim=2, mode="c"] outputs not None):

    cdef int ierr

    # check dimensions
    if inputs.shape[0] != outputs.shape[0]:
        raise ValueError("inputs.shape[0] != outputs.shape[0]")

    ierr = c_gr2m_update_run(inputs.shape[0], \
            config.shape[0], \
            params.shape[0], \
            inputs.shape[1], \
            statesini.shape[0], \
            outputs.shape[1], \
            start, end, \
            <double*> np.PyArray_DATA(config), \
            <double*> np.PyArray_DATA(params), \
            <double*> np.PyArray_DATA(inputs), \
            <double*> np.PyArray_DATA(statesini), \
            <double*> np.PyArray_DATA(outputs))

    return ierr


