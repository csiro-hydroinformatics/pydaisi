#include "c_gr2m_update.h"

/* Utility to get the constant in python */
int c_get_update_params_start() {
    return GR2M_UPDATE_PARAMS_START;
}


/*******************************************************************************
* Run time step code for the MODIF rainfall-runoff model
*
* --- Inputs
* ierr			Error message
* nconfig		Number of configuration elements (3)
* nparams		Number of params (33: 27 for S, 9 for R, 3 for P3 and 3 for Q)
* ninputs		Number of inputs (2)
* nstates		Number of states (2, S and R)
*
* configs
* params			Model paramsameters. 1D Array nparams(4)x1
*					params[0] = S
*					params[1] = IGF
*
* uh			uh ordinates. 1D Array nuhx1
*
* inputs		Model inputs. 1D Array ninputs(2)x1
*
* statesuh		uh content. 1D Array nuhx1
*
* states		Output and states variables. 1D Array nstates(11)x1
*
*******************************************************************************/

int c_gr2m_update_runtimestep(int nconfig, int nparams, int ninputs,
        int nstates, int noutputs,
	    double * config,
	    double * params,
        double * inputs,
        double * states,
        double * outputs)
{
    int ierr=0;
    double P, E;

    /* production variables */
    double phi, psi, S1, S2, Sr, S3, P1, P2;
    double Sstart, AE, Send, P3, Send_nocheck, P3_nocheck;
    double u_update, u_baseline;
    double p3n_update, p3n_baseline;

    /* Routing variables */
    double R1, R2, R3;
    double Rstart, F, Rend, Rend_nocheck;
    double v_update, v_baseline;
    double Q, Q1, qn_update, qn_baseline, Q_nocheck;
    double tPdelta, tEdelta, Sdelta, tP3delta, Rdelta, tQdelta;

    /* GR2M parameters  and radial basis config */
    double X1 = params[0];
    double X2 = params[1];
    double Xr = params[2];

    /* transform parameters */
    double lamP = config[1];
    double lamE = config[2];
    double lamQ = config[3];
    double nu = config[4];

    /* interpolation variables */
    int ibfun, NPT2D=6, NPT3D=10;
    double val1, val2, val3;
    double w2d[6] = {0};
    double w3d[10] = {0};

    /* inputs */
    P = inputs[0] < 0 ? 0 : inputs[0];
    E = inputs[1] < 0 ? 0 : inputs[1];

    /* Perturbations */
    tPdelta  = inputs[2];
    tEdelta  = inputs[3];
    Sdelta   = inputs[4];
    tP3delta = inputs[5];
    Rdelta   = inputs[6];
    tQdelta  = inputs[7];

    /* Perturb states AT THE BEGINNING OF THE TIME STEP !!! */
    Sstart = c_minmax(0, X1, states[0]+Sdelta);
    Rstart = c_minmax(0, Xr, states[1]+Rdelta);

    /* .. perturb inputs - always clip to 0 */
    P = c_boxcox_perturb(P, tPdelta, lamP, nu, 0);
    E = c_boxcox_perturb(E, tEdelta, lamE, nu, 0);

    /*** PRODUCTION ***/

    /* orignal GR2M */
    phi = tanh(P/X1);
    psi = tanh(E/X1);
    S1 = (Sstart+X1*phi)/(1+phi*Sstart/X1);
    P1 = P+Sstart-S1;
    S2 = S1*(1-psi)/(1+psi*(1-S1/X1));
    Sr = S2/X1;
    S3 = S2/cbrt(1.+Sr*Sr*Sr);
    P2 = S2-S3;

    u_baseline = c_gr2m_prod_S_raw2norm(X1, S3);
    p3n_baseline = c_gr2m_prod_P3_raw2norm(X1, P1+P2);

    /* Set 3D basis in interpolation */
    val1 = c_gr2m_prod_S_raw2norm(X1, Sstart);
    val2 = c_gr2m_prod_P_raw2norm(X1, P);
    val3 = c_gr2m_prod_E_raw2norm(X1, E);
    w3d[0] = 1.;
    w3d[1] = val1;
    w3d[2] = val2;
    w3d[3] = val3;
    w3d[4] = val1*val1;
    w3d[5] = val2*val2;
    w3d[6] = val3*val3;
    w3d[7] = val1*val2;
    w3d[8] = val1*val3;
    w3d[9] = val2*val3;

    for(ibfun=10; ibfun<NPT3D; ibfun++)
        w3d[ibfun] = 0.;

    /* S reservoir */
    /* .. approx */
    u_update = 0;
    for(ibfun=0; ibfun<NPT3D; ibfun++) {
        u_update += w3d[ibfun]*config[GR2M_UPDATE_PARAMS_START+ibfun];
    }

    /* .. back transform */
    Send = c_gr2m_prod_S_norm2raw(X1, u_baseline+u_update);

    /* .. check bounds */
    Send_nocheck = Send;

    /* P3 value */
    /* .. approx */
    p3n_update = 0;
    for(ibfun=0; ibfun<NPT3D; ibfun++)
        p3n_update += w3d[ibfun]*config[GR2M_UPDATE_PARAMS_START+NPT3D+ibfun];

    /* .. back transform */
    P3 = c_gr2m_prod_P3_norm2raw(X1, p3n_baseline+p3n_update);
    P3_nocheck = P3;

    /* .. check bounds and mass balance (max = assumes 0 AET) */
    Send = c_minmax(0, c_min(Sstart+P, X1), Send);
    P3 = c_minmax(0, P+Sstart-Send, P3);

    /* .. perturb */
    P3 = c_boxcox_perturb(P3, tP3delta, lamP, nu, 0.);

    /* .. computes AE from mass balance residual */
    AE = P+Sstart-Send-P3;

    /*** ROUTING ***/

    /* Original GR2M */
    R1 = Rstart+P3;
    R2 = X2*R1;
    Q1 = R2*R2/(R2+Xr);
    R3 = R2-Q1;

    v_baseline = c_gr2m_rout_Rend_raw2norm(X2, Xr, R3);
    qn_baseline = c_gr2m_rout_Q_raw2norm(X2, Xr, Q1);

    /* 2D interpolation basis */
    val1 = c_gr2m_rout_Rstart_raw2norm(X2, Xr, Rstart);
    val2 = c_gr2m_rout_P3_raw2norm(X2, Xr, P3);
    w2d[0] = 1.;
    w2d[1] = val1;
    w2d[2] = val2;
    w2d[3] = val1*val1;
    w2d[4] = val2*val2;
    w2d[5] = val1*val2;

    for(ibfun=6; ibfun<NPT2D; ibfun++)
        w2d[ibfun] = 0.;

    /* R reservoir */
    /* .. approx */
    v_update = 0;
    for(ibfun=0; ibfun<NPT2D; ibfun++)
        v_update += w2d[ibfun]*config[GR2M_UPDATE_PARAMS_START+NPT3D*2+ibfun];

    /* .. denormalise .. */
    Rend = c_gr2m_rout_Rend_norm2raw(X2, Xr, v_baseline+v_update);
    Rend_nocheck = Rend;

    /* Q value */
    /* .. approx */
    qn_update = 0;
    for(ibfun=0; ibfun<NPT2D; ibfun++)
        qn_update += w2d[ibfun]*config[GR2M_UPDATE_PARAMS_START+NPT3D*2+NPT2D+ibfun];

    /* .. denormalise .. */
    Q = c_gr2m_rout_Q_norm2raw(X2, Xr, qn_baseline+qn_update);
    Q_nocheck = Q;

    /* .. check bounds */
    Rend = c_minmax(0, Xr, Rend);
    Q = c_max(0, Q);

    /* .. perturb */
    Q = c_boxcox_perturb(Q, tQdelta, lamQ, nu, 0.);

    /* .. computes F from mass balance residual */
    F = Rend-Rstart-P3+Q;

    /* states */
    states[0] = Send;
    states[1] = Rend;

    /* output */
    outputs[0] = Q;

    if(noutputs>1)
        outputs[1] = Send;
    else
        return ierr;

    if(noutputs>2)
        outputs[2] = Rend;

    if(noutputs>3)
        outputs[3] = F;
    else
        return ierr;

    if(noutputs>4)
        outputs[4] = P3;
    else
        return ierr;

    if(noutputs>5)
        outputs[5] = AE;
    else
        return ierr;

    if(noutputs>6)
        outputs[6] = P;
    else
        return ierr;

    if(noutputs>7)
        outputs[7] = E;
    else
        return ierr;

    if(noutputs>8)
        outputs[8] = Send_nocheck;
    else
        return ierr;

    if(noutputs>9)
        outputs[9] = P3_nocheck;
    else
        return ierr;

    if(noutputs>10)
        outputs[10] = Rend_nocheck;
    else
        return ierr;

    if(noutputs>11)
        outputs[11] = Q_nocheck;
    else
        return ierr;

    if(noutputs>12)
        outputs[12] = u_baseline;
    else
        return ierr;

    if(noutputs>13)
        outputs[13] = p3n_baseline;
    else
        return ierr;

    if(noutputs>14)
        outputs[14] = v_baseline;
    else
        return ierr;

    if(noutputs>15)
        outputs[15] = qn_baseline;
    else
        return ierr;

    if(noutputs>16)
        outputs[16] = u_update;
    else
        return ierr;

    if(noutputs>17)
        outputs[17] = p3n_update;
    else
        return ierr;

    if(noutputs>18)
        outputs[18] = v_update;
    else
        return ierr;

    if(noutputs>19)
        outputs[19] = qn_update;
    else
        return ierr;


    return ierr;
}


// --------- Component runner --------------------------------------------------
int c_gr2m_update_run(int nval,
    int nconfig,
    int nparams,
    int ninputs,
    int nstates,
    int noutputs,
    int start, int end,
    double * config,
    double * params,
    double * inputs,
    double * statesini,
    double * outputs)
{
    int ierr=0, i;

    /* Check dimensions */
    if(nconfig != GR2M_UPDATE_NCONFIG)
        return GR2M_UPDATE_ERROR + __LINE__;

    if(nparams != GR2M_UPDATE_NPARAMS)
        return GR2M_UPDATE_ERROR + __LINE__;

    if(nstates != GR2M_UPDATE_NSTATES)
        return GR2M_UPDATE_ERROR + __LINE__;

    if(ninputs != GR2M_UPDATE_NINPUTS)
        return GR2M_UPDATE_ERROR + __LINE__;

    if(noutputs > GR2M_UPDATE_NOUTPUTS)
        return GR2M_UPDATE_ERROR + __LINE__;

    if(start < 0)
        return GR2M_UPDATE_ERROR + __LINE__;

    if(end >=nval)
        return GR2M_UPDATE_ERROR + __LINE__;

    /* Run timeseries */
    for(i = start; i <= end; i++)
    {
        //fprintf(stdout, "\n[%3d]\n", i);
    	ierr = c_gr2m_update_runtimestep(nconfig, nparams,
                ninputs,
                nstates,
                noutputs,
                config,
    		    params,
                &(inputs[ninputs*i]),
                statesini,
                &(outputs[noutputs*i]));

        if(ierr > 0 )
            return ierr;
    }

    return ierr;
}

