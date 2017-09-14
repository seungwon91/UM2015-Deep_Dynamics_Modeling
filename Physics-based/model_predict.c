#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <errno.h>

#include "mex.h"

#ifdef MX_API_VER
#if MX_API_VER < 0x07030000
typedef int mwIndex;
#endif 
#endif 
#ifndef max
#define max(x,y) (((x)>(y))?(x):(y))
#endif
#ifndef min
#define min(x,y) (((x)<(y))?(x):(y))
#endif
#ifndef abs
#define abs(x) ( (x>0)?(x):(-x) )
#endif
#ifndef sign_manual
#define sign_manual(x) ( (x>=0)?(1):(-1) )
#endif

void exit_with_help(){
	mexPrintf(
	"Number of Input/Output Argument is Wrong\n"
	);
}

void print_matrix(double *in_mtx, mwSize nrows, mwSize ncols){
    int cnt_r, cnt_c;
    
    for(cnt_r=0; cnt_r<nrows; cnt_r++){
        for(cnt_c=0; cnt_c<ncols; cnt_c++){
            mexPrintf("%f\t", in_mtx[cnt_r + cnt_c * nrows]);
        }
        mexPrintf("\n");
    }
    return;
}

// copy value of matrix to other(out_mtx = in_mtx)
void matrix_assignment(double *in_mtx, double *out_mtx, mwSize nrows, mwSize ncols){
    int cnt_r, cnt_c;
    
    for(cnt_r=0; cnt_r<nrows; cnt_r++){
        for(cnt_c=0; cnt_c<ncols; cnt_c++){
            out_mtx[cnt_r + cnt_c*nrows] = in_mtx[cnt_r + cnt_c*nrows];
        }
    }
    return;
}

// out_matrix = in_matrix1 + in_matrix2
void matrix_addition(double *in_matrix1, double *in_matrix2, double *out_matrix, mwSize nrows, mwSize ncols){
    int cnt_r, cnt_c;
    
    for(cnt_r=0; cnt_r<nrows; cnt_r++){
        for(cnt_c=0; cnt_c<ncols; cnt_c++){
            out_matrix[cnt_r + cnt_c*nrows] = in_matrix1[cnt_r + cnt_c*nrows] + in_matrix2[cnt_r + cnt_c*nrows];
        }
    }
    return;
}

// out_matrix = in_scalar * in_matrix
void matrix_scalar_multiplication(double *in_matrix, double in_scalar, double *out_matrix, mwSize nrows, mwSize ncols){
    int cnt_r, cnt_c;
    
    for(cnt_r=0; cnt_r<nrows; cnt_r++){
        for(cnt_c=0; cnt_c<ncols; cnt_c++){
            out_matrix[cnt_r + cnt_c*nrows] = in_scalar * in_matrix[cnt_r + cnt_c*nrows];
        }
    }
    return;
}

// out_matrix = in_matrix1 * in_matrix2
void matrix_multiplication(double *in_matrix1, double *in_matrix2, double *out_matrix, mwSize nrows, mwSize ncols, mwSize nelems){
    int cnt, cnt_r, cnt_c;
    double tmp;

    for(cnt_r=0; cnt_r<nrows; cnt_r++){
        for(cnt_c=0; cnt_c<ncols; cnt_c++){
            tmp = 0;
            for(cnt=0; cnt<nelems; cnt++){
                tmp += in_matrix1[cnt_r + cnt*nrows] * in_matrix2[cnt + cnt_c*nelems];
            }
            out_matrix[cnt_r + cnt_c*nrows] = tmp;
        }
    }
    return;
}

// model of motor subsystem in physics-based model of wheelchair(Quantum6000z)
void motor_model_c(double *motor_state, double *friction_out, double motor_input,
        double delta_t, double param_mu, double param_beta, double param_gamma, double param_alpha){
    double friction_threshold;
    mxArray *mtx_A, *mtx_Bf, *mtx_Bd, *mtx_tmp1, *mtx_tmp2, *mtx_tmp3;
    double *mtx_A_ptr, *mtx_Bf_ptr, *mtx_Bd_ptr, *mtx_tmp1_ptr, *mtx_tmp2_ptr, *mtx_tmp3_ptr;
    
    // parameter matrix set-up
    mtx_A = mxCreateDoubleMatrix(2, 2, mxREAL);
    mtx_A_ptr = mxGetPr(mtx_A);
    mtx_A_ptr[0] = 1;
    mtx_A_ptr[1] = -param_beta * delta_t;
    mtx_A_ptr[2] = delta_t;
    mtx_A_ptr[3] = 1 - param_gamma * delta_t;
    
    mtx_Bf = mxCreateDoubleMatrix(2, 1, mxREAL);
    mtx_Bf_ptr = mxGetPr(mtx_Bf);
    mtx_Bf_ptr[0] = delta_t;
    mtx_Bf_ptr[1] = 0;
    
    mtx_Bd = mxCreateDoubleMatrix(2, 1, mxREAL);
    mtx_Bd_ptr = mxGetPr(mtx_Bd);
    mtx_Bd_ptr[0] = 0;
    mtx_Bd_ptr[1] = param_alpha * delta_t;
    
    // temporary variable set-up
    mtx_tmp1 = mxCreateDoubleMatrix(2, 1, mxREAL);
    mtx_tmp1_ptr = mxGetPr(mtx_tmp1);
    mtx_tmp2 = mxCreateDoubleMatrix(2, 1, mxREAL);
    mtx_tmp2_ptr = mxGetPr(mtx_tmp2);
    mtx_tmp3 = mxCreateDoubleMatrix(2, 1, mxREAL);
    mtx_tmp3_ptr = mxGetPr(mtx_tmp3);
    
    // friction threshold set-up
    friction_threshold = -motor_state[0]/delta_t - motor_state[1];
    if(abs(friction_threshold) <= param_mu)
        friction_out[0] = friction_threshold;
    else
        friction_out[0] = sign_manual(friction_threshold) * param_mu;
    
    // update motor state(motor_state = A*motor_state + B_f*friction_out + B_d*motor_input)
    matrix_multiplication(mtx_A_ptr, motor_state, mtx_tmp1_ptr, 2, 1, 2);
        //matrix_scalar_multiplication(mtx_Bf_ptr, friction_out, mtx_tmp2_ptr, 2, 1);
    matrix_multiplication(mtx_Bf_ptr, friction_out, mtx_tmp2_ptr, 2, 1, 1);
    matrix_addition(mtx_tmp1_ptr, mtx_tmp2_ptr, mtx_tmp3_ptr, 2, 1);
    
    matrix_scalar_multiplication(mtx_Bd_ptr, motor_input, mtx_tmp2_ptr, 2, 1);
    matrix_addition(mtx_tmp2_ptr, mtx_tmp3_ptr, mtx_tmp1_ptr, 2, 1);
    
    matrix_assignment(mtx_tmp1_ptr, motor_state, 2, 1);
    return;
}

// Function to make desired output of this c file from inputs
void predict_phys(mxArray *plhs[], mxArray *prhs[]){
	double muR, muL, betaR, betaL, gammaR, gammaL, alphaR, alphaL, turnRate, turnRateInPlace, turnReductionRate, wheelbase;
    double dt, turnRateModified;
    double *param_array_ptr, *data_input_ptr; // pointer for input matrices
    double *vs_ptr, *omegas_ptr, *us_joystick_ptr, *qs_ptr, *us_motor_ptr, *fs_ptr; // pointer for output matrices
    double *q_l_ptr, *q_r_ptr, *u_joystick_ptr, *u_motor_ptr, *input_trans_mtx_ptr, *u_friction_r_ptr, *u_friction_l_ptr;
	mxArray *q_l, *q_r, *u_joystick, *u_motor, *input_trans_mtx, *u_friction_r, *u_friction_l;
    int num_data_cases, num_pred_step, data_cnt, pred_cnt;

    // model parameter set-up
    param_array_ptr = mxGetPr(prhs[0]);
	muR = param_array_ptr[0];
	betaR = param_array_ptr[1];
	gammaR = param_array_ptr[2];
	alphaR = param_array_ptr[3];
	muL = param_array_ptr[4];
	betaL = param_array_ptr[5];
	gammaL = param_array_ptr[6];
	alphaL = param_array_ptr[7];
	turnRate = param_array_ptr[8];
	turnRateInPlace = param_array_ptr[8];
	turnReductionRate = param_array_ptr[9];
    wheelbase = param_array_ptr[10];
    
    // dealing with other input arguments
    num_data_cases = mxGetM(prhs[1]);
    data_input_ptr = mxGetPr(prhs[1]);
    num_pred_step = mxGetScalar(prhs[2]);
    
    // output variable set-up
    plhs[0] = mxCreateDoubleMatrix(num_data_cases, num_pred_step, mxREAL);
    vs_ptr = mxGetPr(plhs[0]);
    plhs[1] = mxCreateDoubleMatrix(num_data_cases, num_pred_step, mxREAL);
    omegas_ptr = mxGetPr(plhs[1]);
    plhs[2] = mxCreateDoubleMatrix(num_data_cases, 2*num_pred_step, mxREAL);
    us_joystick_ptr = mxGetPr(plhs[2]);
    plhs[3] = mxCreateDoubleMatrix(num_data_cases, 2*num_pred_step, mxREAL);
    qs_ptr = mxGetPr(plhs[3]);
    plhs[4] = mxCreateDoubleMatrix(num_data_cases, 2*num_pred_step, mxREAL);
    us_motor_ptr = mxGetPr(plhs[4]);
    plhs[5] = mxCreateDoubleMatrix(num_data_cases, 2*num_pred_step, mxREAL);
    fs_ptr = mxGetPr(plhs[5]);
    
    // intermediate variable set-up
    dt = (double)1/25;
    q_l = mxCreateDoubleMatrix(2, 1, mxREAL);
    q_l_ptr = mxGetPr(q_l);
    q_r = mxCreateDoubleMatrix(2, 1, mxREAL);
    q_r_ptr = mxGetPr(q_r);
    u_joystick = mxCreateDoubleMatrix(2, 1, mxREAL);
    u_joystick_ptr = mxGetPr(u_joystick);
    u_motor = mxCreateDoubleMatrix(2, 1, mxREAL);
    u_motor_ptr = mxGetPr(u_motor);
    u_friction_r = mxCreateDoubleMatrix(1, 1, mxREAL);
    u_friction_r_ptr = mxGetPr(u_friction_r);
    u_friction_l = mxCreateDoubleMatrix(1, 1, mxREAL);
    u_friction_l_ptr = mxGetPr(u_friction_l);
    input_trans_mtx = mxCreateDoubleMatrix(2, 2, mxREAL);
    input_trans_mtx_ptr = mxGetPr(input_trans_mtx);
    input_trans_mtx_ptr[0] = 1;
    input_trans_mtx_ptr[1] = 1;
    
    for(data_cnt=0; data_cnt<num_data_cases; data_cnt++){
        // get initial state from input data matrix
        q_l_ptr[0] = data_input_ptr[data_cnt];
        q_l_ptr[1] = data_input_ptr[data_cnt+num_data_cases];
        q_r_ptr[0] = data_input_ptr[data_cnt+2*num_data_cases];
        q_r_ptr[1] = data_input_ptr[data_cnt+3*num_data_cases];
        
        for(pred_cnt=0; pred_cnt<num_pred_step; pred_cnt++){
            u_joystick_ptr[0] = data_input_ptr[data_cnt+(2*pred_cnt+4)*num_data_cases];
            u_joystick_ptr[1] = data_input_ptr[data_cnt+(2*pred_cnt+5)*num_data_cases];
            
            turnRateModified = turnRate*(1-turnReductionRate*u_joystick_ptr[0]);
            if(u_joystick_ptr[0]<0.00001 && u_joystick_ptr[0]>-0.00001){
                input_trans_mtx_ptr[2] = turnRateInPlace;
                input_trans_mtx_ptr[3] = -turnRateInPlace;
            }
            else{
                input_trans_mtx_ptr[2] = turnRateModified;
                input_trans_mtx_ptr[3] = -turnRateModified;
            }
            matrix_multiplication(input_trans_mtx_ptr, u_joystick_ptr, u_motor_ptr, 2, 1, 2);
            
            // use motor model to update motor states
            motor_model_c(q_r_ptr, u_friction_r_ptr, u_motor_ptr[0], dt, muR, betaR, gammaR, alphaR);
            motor_model_c(q_l_ptr, u_friction_l_ptr, u_motor_ptr[1], dt, muL, betaL, gammaL, alphaL);
            
            // save result
            vs_ptr[data_cnt+pred_cnt*num_data_cases] = (q_r_ptr[0] + q_l_ptr[0])*0.5;
            
            omegas_ptr[data_cnt+pred_cnt*num_data_cases] = (q_r_ptr[0] - q_l_ptr[0])/wheelbase;
            
            us_joystick_ptr[data_cnt+2*pred_cnt*num_data_cases] = u_joystick_ptr[0];
            us_joystick_ptr[data_cnt+(2*pred_cnt+1)*num_data_cases] = u_joystick_ptr[1];
            
            qs_ptr[data_cnt+2*pred_cnt*num_data_cases] = q_l_ptr[0];
            qs_ptr[data_cnt+(2*pred_cnt+1)*num_data_cases] = q_r_ptr[0];
            
            us_motor_ptr[data_cnt+2*pred_cnt*num_data_cases] = u_motor_ptr[0];
            us_motor_ptr[data_cnt+(2*pred_cnt+1)*num_data_cases] = u_motor_ptr[1];
            
            fs_ptr[data_cnt+2*pred_cnt*num_data_cases] = u_friction_l_ptr[0];
            fs_ptr[data_cnt+(2*pred_cnt+1)*num_data_cases] = u_friction_r_ptr[0];
        }
    }
    return;
}


// Input : parameters(input-mapping, motor-subsystem, and wheelbase), data_x, num_simulation_step
// Output : vs(linear velocity), omegas(angular velocity),
//          us_joystick(joystick command), qs(wheel speed left, right),
//          us_motor(motor input), fs(friction of each motor)
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] ){

	if(nrhs != 3 || nlhs != 6)
	{
		exit_with_help();
		return;
	}

    mexPrintf("Using C code to make 5 seconds simulation\n");
	predict_phys(plhs, prhs);

	return;
}