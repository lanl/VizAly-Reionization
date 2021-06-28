import numpy as np
import math
import enum

# Using same tolerance as in radiation.F90, line 623 of C2-Ray3Dm: 
# https://github.com/garrelt/C2-Ray3Dm/blob/c181c44c7b5fea7f6ee256e4358e5cadff98dce1/radiation.F90#L623
thin_tol = 1e-2 
# thin_tol = 1e-5

def yr2sec(y):
    # return 3.154e7*y
    return 31557600*y # julian year, as used in astronomy
    # return 3.1558149e7*y


# LERP. Assumes interpolation and not extrapolation
def lerp(a, b, t):
    assert (0 <= t) and (t <= 1)
    return a*(1-t) + t*b

# L is the result of the LERP, with unknown parameter t.
# Assumes interpolation, NOT extrapolation, and assumes that (b - a) is
# nonzero (numerically, it cannot be too close to zero)
# returns t
def inv_lerp(a, b, L):
    assert not math.isclose(a,b)
    t = (L - a)/(b - a) 
    assert (0 <= t) and (t <= 1)
    return t
    
# MODIFIED version of eq. 5 in Mella et. al 2006
def get_V_shell(r, Delta_r):
    return (4/3)*np.pi*((r + Delta_r)**3 - r**3)

# C_H is the collisional ionization coefficient
# C is the clumping factor
def get_x_eq(Gamma, n_e, C_H, C, alpha_H): # equilibrium (a.k.a. steady-state) solution
    return (Gamma + n_e*C_H)/(Gamma + n_e*C_H + n_e*C*alpha_H) # eq. 14, steady-state solution

# C_H is the collisional ionization coefficient
# C is the clumping factor
def get_t_i(Gamma, n_e, C_H, C, alpha_H): # characteristic time
    denominator = Gamma + n_e*C_H + n_e*C*alpha_H
    # print(denominator)
    return 1/denominator # eq. 13

# This is the constant of proportionality implied in eq. 22. Assumes gray
# opacity. 
#
# See my handwritten notes for derivation of this coefficient. I have separate
# derivations for both the optically thin and thick cases.
#
# For the optically thin case, my assumption is that both σ and Δr have
# nontrivial values - i.e. I am taking the limit as n_HI --> 0, keeping σ and
# Δr fixed.
def attenuation_coeff(tau, Del_tau, n_H, n_HI, sigma, Del_r, V_shell):
    if Del_tau < thin_tol: # optically thin
    # if n_HI < n_HI_tol*n_H: # optically thin
        return (np.exp(-tau)*sigma*Del_r) / V_shell
    return np.exp(-tau)*(1 - np.exp(-Del_tau)) / (n_HI*V_shell)

# ***************************************
# Diff eq. solve functions (for both linear and nonlinear cases) below. 
# TODO: Incorporate clumping parameter C, not to be confused with collisional
# ionization coefficient C_H
# ***************************************

# Solution to ODE in eq. 11, assuming FIXED Γ, n_e C_H, α_H. 
# Also assuming C = 1
# If the aforementioned assumption is true, the ODE is linear and 
# there is an analytic solution.
def x_evolve(t, x_0, Gamma, n_e, C_H, alpha_H): 
    C = 1
    t_i = get_t_i(Gamma, n_e, C_H, C, alpha_H) 
    x_eq = get_x_eq(Gamma, n_e, C_H, C, alpha_H)
    return x_eq + (x_0 - x_eq)*np.exp(-t/t_i)

def listdict_insert(listdict, key, val):
    if not (key in listdict):
        listdict[key] = []
    listdict[key].append(val)

def log_iter(log_dict, step_info, iter_num, step_num):
    listdict_insert(log_dict, 'iter_num', iter_num)
    listdict_insert(log_dict, 'step_num', step_num)
    for key, val in step_info.items():
        listdict_insert(log_dict, key, val)

# Performs a single iteration of the nonlinear solve for the time-averaged ionization fraction,
# as depicted in the flowchart in Figure 2, p. 378, Mellema et al 2006
def x_nonlinear_iter(x0_i, N_HI_to_cell, V_shell, x_avg, step_info): 
    n_H = step_info['n_H']
    sigma = step_info['sigma']
    Del_r = step_info['Del_r']
    col_density = N_HI_to_cell
    dot_N_gamma = step_info['dot_N_gamma']
    Del_t = step_info['Del_t']
    C_H = step_info['C_H']
    C = step_info['C']
    alpha_H = step_info['alpha_H']

    Del_tau_avg = (1 - x_avg)*n_H*sigma*Del_r # eq. 16 ... I assume GRAY opacity
    n_HI_avg = (1 - x_avg)*n_H
    attenuation = attenuation_coeff(sigma*col_density, Del_tau_avg, n_H, n_HI_avg, sigma, Del_r, V_shell)
    step_info['attenuation'] = attenuation  # for debug log
    Gamma = attenuation*dot_N_gamma # eq. 17
    step_info['gamma'] = Gamma # for debug log
    n_e_avg = x_avg*n_H 

    # Gamma_tol = 1e-30
    # n_e_tol = 1e-30
    # if Gamma < Gamma_tol and n_e_avg < n_e_tol: # denominator in eqn. 13 and 14 close to zero
    #     step_info['inf_ti'] = 1.0
    #     return x_avg # return same x_avg as before because dx/dt ≈ 0
    # else:
    #     step_info['inf_ti'] = 0.0

    denom = Gamma + n_e_avg*C_H + n_e_avg*C*alpha_H
    if denom < 1/step_info['t_i_tol']:
        step_info['inf_ti'] = 1.0
        return x_avg # return same x_avg as before because dx/dt ≈ 0
    else: 
        step_info['inf_ti'] = 0.0

    t_i = get_t_i(Gamma, n_e_avg, C_H, C, alpha_H) 

    x_eq = np.clip(get_x_eq(Gamma, n_e_avg, C_H, C, alpha_H), 0.0, 1.0)
    step_info['t_i'] = t_i
    step_info['x_eq'] = x_eq

    new_x_avg = x_eq + (x0_i - x_eq)*(1 - np.exp(-Del_t/t_i))*(t_i/Del_t) # eq.15
    new_x_avg = np.clip(new_x_avg, 0.0, 1.0)
    if math.isnan(new_x_avg):
        raise Exception("NaN encountered!")
    return new_x_avg

def x_nonlinear_solve(cell_idx, x0_i, N_HI_to_cell, V_shell, step_info, step_num, log_dict):
    x_arr = step_info['x_arr']

    x_prev = float('inf')
    iter_count = 0
    x_avg_curr = x_arr[cell_idx] # for debugging
    while abs(x_prev - x_avg_curr) > step_info['tol']:
        # if cell_idx == 17: # for debug
        #     print(x_avg_curr)
        # log_iter(log_dict, step_info, iter_count, step_num) 

        x_prev = x_avg_curr
        x_avg_curr = x_nonlinear_iter(x0_i, N_HI_to_cell, V_shell, x_avg_curr, step_info)

        # for debug / visualization of convergence
        if step_info['do_log']:
            curr_slice = np.copy(x_arr[0:step_info['log_slice_len']])
            curr_slice[cell_idx] = x_avg_curr
            step_info['log'].append((cell_idx, iter_count, curr_slice))
            log_iter(log_dict, step_info, iter_count, step_num) 
    
        iter_count += 1
        if iter_count > 100:
            raise Exception("Failing to converge at cell %d" % cell_idx)
    return x_avg_curr

def step(x_0, step_num, step_info, num_cells, tau_log, log_dict):
    x_arr = step_info['x_arr']
    Del_r = step_info['Del_r']
    n_H = step_info['n_H']
    N_HI_to_cell = 0 # column density to cell
    for i in range(num_cells): # i is the cell index
        r = i*Del_r
        V_shell = get_V_shell(r, Del_r)

        step_info['x_0i'] = x_0[i]
        x_arr[i] = x_nonlinear_solve(i, x_0[i], N_HI_to_cell, V_shell, step_info, step_num, log_dict)

        N_HI_to_cell += n_H*(1-x_arr[i])*Del_r # add col. density contrib. of this cell
        if step_info['do_log']:
            tau_log.append(np.exp(-N_HI_to_cell*step_info['sigma']))
    if step_info['do_log']:
        curr_slice = np.copy(x_arr[0:step_info['log_slice_len']])
        listdict_insert(step_info, 'step_log', curr_slice)
    step_info['field_log'].append(np.copy(x_arr)) 

class FltPrecision(enum.Enum):
  Single = 0x1
  Double = 0x2
  LongDouble = 0x3

def find_r_I(x_arr, x_pos_arr, thresh=0.5):
    b_idx = 0
    while(x_arr[b_idx] > thresh):
        b_idx += 1
    a_idx = b_idx - 1
    a = x_pos_arr[a_idx]
    b = x_pos_arr[b_idx]
    x_a = x_arr[a_idx]
    x_b = x_arr[b_idx]
    r_I = lerp(a, b, inv_lerp(x_a, x_b, thresh))
    return r_I

def run_all_timesteps(step_info, num_cells, t_evol, tau_log, log_dict, flt_precision, x_pos_arr):
    # x_0 = np.zeros(num_cells) # starting ionization fraction is 0 everywhere!

    if flt_precision == FltPrecision.Double:
        desired_type = np.float64
    elif flt_precision == FltPrecision.Single:
        desired_type = np.float32
    elif flt_precision == FltPrecision.LongDouble:
        desired_type = np.longdouble
        
    x_0 = np.zeros(num_cells, dtype=desired_type)
    # HACK / for debug purposes: change parameters inside step_info to desired floating-point precision
    for key in step_info:
        if isinstance(step_info[key], float):
            step_info[key] = desired_type(step_info[key])

    if step_info['log_ifront']:
        log_dict['ifront_pos'] = []
        log_dict['t_sample'] = [] # for debug

    step_info['x_arr'] = np.copy(x_0)
    curr_t = desired_type(0.0)
    t_since_last_out = desired_type(0.0)
    step_num = 0
    first_step = True
    while curr_t <= t_evol:

        if step_info['log_ifront'] and t_since_last_out > step_info['t_bw_steps']:
            if first_step:
                curr_r_I = desired_type(0.0)
            else:
                curr_r_I = find_r_I(step_info['x_arr'], x_pos_arr) # assume threshold of 0.5 for now
            log_dict['ifront_pos'].append(curr_r_I)
            log_dict['t_sample'].append(curr_t) # for debug
            t_since_last_out = desired_type(0.0)

        # print("Step: %d, time: %f" % (step_num,curr_t))
        step(x_0, step_num, step_info, num_cells, tau_log, log_dict)
        x_0 = np.copy(step_info['x_arr'])
        curr_t += step_info['Del_t']
        t_since_last_out += step_info['Del_t']
        step_num += 1
        if first_step:
            first_step = False
