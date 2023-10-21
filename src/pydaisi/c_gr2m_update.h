#ifndef __GR2M_UPDATE__
#define __GR2M_UPDATE__

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "c_daisi_utils.h"

/* Number of config required by update run
    update coefs:
    10 -> S
    10 -> P3
    6 -> R
    6 -> Q
    ------
    32
    + 5 -> Xr, lamP, lamE, lamQ, nu
    ------
    37
*/
#define GR2M_UPDATE_NCONFIG 37
#define GR2M_UPDATE_PARAMS_START 5

/* Number of inputs required by update run :
  P, E, perturb P, perturb PET, P3, S, R, Q */
#define GR2M_UPDATE_NINPUTS 8

/* X1, X2 + Xr added as extra params */
#define GR2M_UPDATE_NPARAMS 5

/* Number of states returned by update run */
#define GR2M_UPDATE_NSTATES 2

/* Number of outputs returned by update run */
#define GR2M_UPDATE_NOUTPUTS 20

int c_get_modif_params_start();

int c_gr2m_update_run(int nval, int nconfig, int nparams, int ninputs,
    int nstates, int noutputs,
    int start, int  end,
	double * config,
	double * params,
	double * inputs,
	double * statesini,
    double * outputs);

#endif
