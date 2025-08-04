import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from lmfit import minimize, Parameters
from scipy.integrate import solve_ivp
import scipy.integrate as scint
import datetime
from bayes_opt import BayesianOptimization, acquisition
import warnings
warnings.filterwarnings("ignore")

# Define the system of differential equations
def bee_eq(t, y, w1,w2, env_temp_t, h, solar_t, 
           k_W, k_A_W, k_hive, eta_W, eta_A_W, eta_B_W, theta_ideal,
           P_h, P_c, theta_h, theta_c, k_c, k_h, C_c, C_h):
    hive_temp, theta_A_w, theta_B_W, theta_S_w = y
    u1 = 0
    u2 = 0

    if hive_temp < theta_h and theta_B_W <= heat_limit:
        u1 = P_h/C_h
    elif hive_temp > theta_c and theta_A_w >= cool_limit:
        u2 = -P_c/C_c
            
    env_temp = env_temp_t(t) #environment temperature
    solar = solar_t(t) #solar radiation

    A_W_w = (k_A_W+k_c)*(-theta_A_w+env_temp)+eta_A_W*solar+k_hive*(-theta_A_w+hive_temp)
    B_W_w = (k_W+k_h)*(-theta_B_W+env_temp)+eta_B_W*solar+k_hive*(-theta_B_W+hive_temp)
    S_W_w = k_W*(-theta_S_w+env_temp)+eta_W*solar+k_hive*(-theta_S_w+hive_temp)
    M = -h*(w1*w2*(1-np.exp(-hive_temp+theta_ideal)) / (w2+w1*np.exp(-hive_temp+theta_ideal)))
    W = 4*(k_W+k_hive)*(-hive_temp + env_temp) + 4*eta_W*solar 
    A_W = (k_A_W+k_c+k_hive)*(-hive_temp + env_temp) + eta_A_W*solar
    B_W = (k_W+k_h+k_hive)*(-hive_temp + env_temp) + eta_B_W*solar
    return [M + W + A_W + B_W + u1 + u2, A_W_w + u2, B_W_w + u1, S_W_w]

def bee_eq_nc(y, t, r_c, r_h, env_temp,h, solar, k_W, k_A_W, k_hive, eta_W, eta_A_W, eta_B_W, theta_ideal):
    hive_temp = y
    M = -h*(r_c*r_h*(1-np.exp(-hive_temp+theta_ideal)) / (r_h+r_c*np.exp(-hive_temp+theta_ideal)))
    W = 4*(k_W+k_hive)*(-hive_temp + env_temp) + 4*eta_W*solar 
    A_W = (k_A_W+k_hive)*(-hive_temp + env_temp) + eta_A_W*solar
    B_W = (k_W+k_hive)*(-hive_temp + env_temp) + eta_B_W*solar
    dydt = M + W + A_W + B_W
    return dydt

def run_bee_eq(t, a, r_c, r_h, env_temp,h, solar, k_W, k_A_W, k_hive, eta_W, eta_A_W, eta_B_W, theta_ideal):
    sol = scint.odeint(bee_eq_nc, a, t, args=(r_c, r_h, env_temp,h, solar, k_W, k_A_W, k_hive, 
                                           eta_W, eta_A_W, eta_B_W, theta_ideal), 
                                           col_deriv = True, rtol = 10e-3, atol = 10e-3) #w' and 'amplitude_temp_ext'
    theta_t = sol[-1,:]
    return theta_t    

def residual(ps, ts, data,l, k_W, k_A_W, k_hive, eta_W, eta_A_W, eta_B_W):
    d = pd.DataFrame(data).groupby(data['Date'])
    model = []
    k = 0
    r_c = 124 #ps['w1'].value
    r_h = 428 #ps['w2'].value
    h_ = [1]
    return_value = []
    for m,n in d:
        h = ps['h_'+str(k)].value
        theta_ideal = ps['theta_'+str(k)].value + 273.15
        env_temp_ = n['Air Temp'] + 273.15
        solar = n['Sol Rad']
        t_max = len(n['Air Temp'])-1
        t = np.linspace(0,t_max, num = t_max+1)
        a = [1]*len(env_temp_)
        fitted = run_bee_eq(t, a, r_c, r_h, env_temp_, h, solar, k_W, k_A_W, k_hive, eta_W, eta_A_W, eta_B_W, theta_ideal)
        model = np.concatenate((model, fitted - 273.15))
        try:
            h_.append(abs(h - ps['h_'+str(k-1)].value))
        except:
            h_.append(abs(h_[k] - h))
        k = k+1
    return_value = np.concatenate((return_value, (model - data['Temp']).ravel()))
    return_value = np.concatenate((return_value, l*np.array(h_[2:]).ravel()))
    return return_value


def fit_values(data_input, k_W, k_A_W, k_hive, eta_W, eta_A_W, eta_B_W):
    params = Parameters()
    data_input_group = data_input.groupby(['Date'])
    a = 0
    for i,j in data_input_group:
        params.add('h_'+str(a), value = 0.5, min = 0.1, max = h_max)
        params.add('theta_'+str(a), value = np.mean(j['Temp']), min = np.min(j['Temp']), max = np.max(j['Temp']))  
        a = a + 1   
    t_max = len(data_input)-1
    t = np.linspace(0,t_max, num = t_max+1)
    result = minimize(residual, params, args=(t,data_input, l, k_W, k_A_W, 
                                              k_hive, eta_W, eta_A_W, eta_B_W),
                                              method='leastsq',nan_policy='omit',max_nfev = 300)
    return result.params['h_6'].value, result.params['theta_6'].value, result.residual[:len(data_input['Temp'])].reshape(data_input['Temp'].shape)

def plot_temp(a_c, a_h, theta_c, theta_h, theta_ideal, fitted_temp, adjust_temp, hive_temp):
    plt.figure(figsize=(15, 3))
    plt.title(f'$a_c=${np.round(a_c, 3)}, $a_h=${np.round(a_h, 3)}')

    plt.plot([0, len(fitted_temp)], [theta_ideal-273.15, theta_ideal-273.15], 'k--', label='Ideal Temp.')
    plt.plot([0, len(fitted_temp)], [theta_c-273.15, theta_c-273.15], 'r--', alpha=0.7, label='$\\theta_c$ and $\\theta_h$')
    plt.plot([0, len(fitted_temp)], [theta_h-273.15, theta_h-273.15], 'r--', alpha=0.7)

    plt.plot(fitted_temp-273.15, 'b', label='Fitted Temp.')
    plt.plot(adjust_temp-273.15, 'g', alpha=0.8, label='Controlled Temp.')
    plt.plot(hive_temp-273.15, 'm', alpha=0.8, label='Hive Temp.')
    plt.fill_between(range(len(fitted_temp)), fitted_temp-273.15, adjust_temp-273.15, color='m', alpha=0.3, label='Reduced Core Temp.')

    plt.legend()
    #plt.ylim(32, 37)

    plt.tight_layout()
    #plt.savefig(file_name)
    plt.show()

def plot_temp_A(a_c, a_h, fitted_peri_A_temp, adjust_peri_A_temp, env_temp):
    plt.figure(figsize=(15, 3))
    plt.title(f'$a_c=${np.round(a_c, 3)}, $a_h=${np.round(a_h, 3)}')

    plt.plot(fitted_peri_A_temp-273.15, 'b', label='Reconstructed Hive Wall (Upper) Temp.')
    plt.plot(adjust_peri_A_temp-273.15, 'g', alpha=0.8, label='Controlled Hive Wall (Upper) Temp.')
    plt.plot(env_temp-273.15, 'm', alpha=0.8, label='Environment Temp.')
    plt.fill_between(range(len(fitted_peri_A_temp)), fitted_peri_A_temp-273.15, adjust_peri_A_temp-273.15, color='m', alpha=0.3, label='Saved Bee Energy for cooling')

    plt.legend()
    #plt.ylim(32, 37)

    plt.tight_layout()
    #plt.savefig(file_name)
    plt.show()

def plot_temp_B(a_c, a_h, fitted_peri_B_temp, adjust_peri_B_temp, env_temp):
    plt.figure(figsize=(15, 3))
    plt.title(f'$a_c=${np.round(a_c, 3)}, $a_h=${np.round(a_h, 3)}')

    plt.plot(fitted_peri_B_temp-273.15, 'b', label='Reconstructed Hive Wall (Lower) Temp.')
    plt.plot(adjust_peri_B_temp-273.15, 'g', alpha=0.8, label='Controlled Hive Wall (Lower) Temp.')
    plt.plot(env_temp-273.15, 'm', alpha=0.8, label='Environment Temp.')
    plt.fill_between(range(len(fitted_peri_B_temp)), fitted_peri_B_temp-273.15, adjust_peri_B_temp-273.15, color='m', alpha=0.3, label='Saved Bee Energy for heating')

    plt.legend()
    #plt.ylim(32, 37)

    plt.tight_layout()
    #plt.savefig(file_name)
    plt.show()

def plot_temp_S(a_c, a_h, fitted_peri_S_temp, adjust_peri_S_temp, env_temp):
    plt.figure(figsize=(15, 3))
    plt.title(f'$a_c=${np.round(a_c, 3)}, $a_h=${np.round(a_h, 3)}')

    plt.plot(fitted_peri_S_temp-273.15, 'b', label='Reconstructed Hive Wall (Lower) Temp.')
    plt.plot(adjust_peri_S_temp-273.15, 'g', alpha=0.8, label='Controlled Hive Wall (Lower) Temp.')
    plt.plot(env_temp-273.15, 'm', alpha=0.8, label='Environment Temp.')
    plt.fill_between(range(len(fitted_peri_S_temp)), fitted_peri_S_temp-273.15, adjust_peri_S_temp-273.15, color='m', alpha=0.3, label='Saved Bee Energy for heating')

    plt.legend()
    #plt.ylim(32, 37)

    plt.tight_layout()
    #plt.savefig(file_name)
    plt.show()

np.random.seed(0)
maximum_power_c = 50
maximum_power_h = 20
heat_limit = 45 + 273.13
cool_limit = 25 + 273.15

data = pd.read_csv('../Full_Data/Control/C265_w_Env.csv')
try:
    for i in range(len(data)):
        current_date = (datetime.datetime.strptime(str(data['Date'].values[i]), "%m/%d/%Y")).date()
        current_time = datetime.datetime.strptime(str(data['Time'].values[i]), "%H:%M:%S").time()
        current_datetime = datetime.datetime.combine(current_date,current_time)
        data['DateTime'][i] = datetime.datetime.strptime(str(current_datetime), "%Y-%m-%d %H:%M:%S")
        data['Date'][i] = data['DateTime'][i].date()
        data['Time'][i] = data['DateTime'][i].time()
except:
    pass

k_W = 0.29 #thermal conductivity of wood -> per hour
eta_W = 0.012*k_W #heat absorption coeff of wood
k_A_W = k_W #thermal conductivity of top surface (wood + aluminum plate) -> per hour
eta_A_W = eta_W #heat absorption coeff of top surface
eta_B_W = eta_W #heat absorption coeff of bottom surface
k_hive = 0.46
k_c = 0.1*k_W #heat absorption coeff of cooler -> randomly assigned
k_h = k_W #heat absorption coeff of heater -> randomly assigned
C_c = 10/3 #2 #heat capacity of cooler -> randomly assigned
C_h = 4/7 #0.7 #heat capacity of heater -> randomly assigned

h_max = 1
l = 96
window_size = 24
BUDGET = 200 #400

all_temp = np.array([[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]).reshape(-1,28)
r_c = 124 #ps['w1'].value
r_h = 428
h, theta_ideal, reconstructed_input = fit_values(data[:24*7], k_W, k_A_W, k_hive, eta_W, eta_A_W, eta_B_W)
theta_ideal += 273.15
reconstructed_input = data[:24*7]['Temp'].values + reconstructed_input
print(h, theta_ideal-273.15)

for i_start in range(24*7, len(data)-window_size, window_size):
    print(f'Start Index: {i_start}')

    ### Step 2: Forecast tomorrow's hive temperature
    data_train = data[i_start:i_start+window_size]
    theta_env_train = data_train['Air Temp'].values + 273.15
    solar_rad_train = data_train['Sol Rad'].values
    hive_temp_train = data_train['Temp'].values + 273.15
    
    T_0 = [reconstructed_input[-1] + 273.15, theta_env_train[0], theta_env_train[0], theta_env_train[0]]
    ts = np.linspace(0, window_size-1, num=window_size)
    t_span = (0, window_size-1)
    t_eval = np.linspace(t_span[0], t_span[1], 24, endpoint=True)  # Time points to evaluate

    t_env_interp_train = interp1d(ts, theta_env_train, kind='slinear', fill_value='extrapolate')
    t_solar_interp_train = interp1d(ts, solar_rad_train, kind='slinear', fill_value='extrapolate')
    
    sol = solve_ivp(bee_eq, t_span, T_0, args = [r_c, r_h, t_env_interp_train, h, t_solar_interp_train, k_W, k_A_W, k_hive, 
                                                       eta_W, eta_A_W, eta_B_W, theta_ideal,
                                                       0, 0, -np.inf, np.inf,
                                                       k_c, k_h, C_c, C_h], t_eval=t_eval, max_step = 0.001, 
                        method= 'LSODA')
    fitted_temp_train = sol.y[0,:]
    fitted_peri_A_train = sol.y[1,:] 
    fitted_peri_B_train = sol.y[2,:]
    fitted_peri_S_train = sol.y[3,:]

    ### Step 3: Optimize controller settings
    #t_c_max = fitted_temp_train.max() - theta_ideal
    #t_h_max = theta_ideal - fitted_temp_train.min()
    buffer_high = theta_ideal + 0.5
    buffer_low = theta_ideal - 0.5
    t_c_max = fitted_temp_train.max() - buffer_high
    t_h_max = buffer_low - fitted_temp_train.min()
    print(buffer_high, buffer_low, t_c_max, t_h_max)

    a_c = maximum_power_c #* params['a_c']
    a_h = maximum_power_h #* params['a_h']

    if t_h_max > 0:
        theta_h = buffer_low 
    else:
        theta_h = -np.nan
        a_h = 0

    ### Maximum temp is higher than ideal temp
    if t_c_max > 0:
        theta_c = buffer_high
    else:
        theta_c = np.inf
        a_c = 0

    T_0 = [reconstructed_input[-1] + 273.15, theta_env_train[0], theta_env_train[0], theta_env_train[0]]

    print(a_h, a_c, theta_ideal, theta_h, theta_c)


    sol = solve_ivp(bee_eq, t_span, T_0, args = [r_c, r_h, t_env_interp_train, h, t_solar_interp_train, k_W, k_A_W, k_hive, 
                                                       eta_W, eta_A_W, eta_B_W, theta_ideal,
                                                       a_h, a_c, theta_h, theta_c,
                                                       k_c, k_h, C_c, C_h], 
                        t_eval=t_eval, max_step = 0.001, 
                        method= 'LSODA')
    adjust_temp_train = sol.y[0,:]
    adjust_peri_A_train = sol.y[1,:] 
    adjust_peri_B_train = sol.y[2,:]
    adjust_peri_S_train = sol.y[3,:]

    total_budget = np.array([float(0)]*len(adjust_temp_train))
    total_cost = np.array([float(0)]*len(adjust_temp_train))
    total_budget[np.where(fitted_temp_train < theta_h)] = a_h
    total_budget[np.where(fitted_temp_train > theta_c)] = a_c

    fitted_train = (fitted_peri_A_train+fitted_peri_B_train+4*fitted_peri_S_train)/6 - fitted_temp_train
    adjusted_train = (adjust_peri_A_train+adjust_peri_B_train+4*adjust_peri_S_train)/6 - adjust_temp_train
    total_cost[np.where(fitted_temp_train < theta_h)] = abs(fitted_train - adjusted_train)[np.where(fitted_temp_train < theta_h)]
    total_cost[np.where(fitted_temp_train > theta_c)] = (r_h/r_c)*abs(fitted_train - adjusted_train)[np.where(fitted_temp_train > theta_c)]
    #arg_sort = np.argsort(-total_cost)

    #print('adjust_peri_B_train before', adjust_peri_B_train)
    selected_points = []
    current_budget = 0
    for i in range(24):
        if current_budget + total_budget[i] <= BUDGET and total_budget[i] != 0:
            current_budget += total_budget[i]
            selected_points.append(i)
        else:
            adjust_temp_train[i] = fitted_temp_train[i]
            if fitted_temp_train[i] > theta_c:
                adjust_peri_A_train[i] = fitted_peri_A_train[i]
            if fitted_temp_train[i] < theta_h:
                adjust_peri_B_train[i] = fitted_peri_B_train[i]

    print('selected_points', selected_points)
    print('adjust_peri_B_train after', adjust_peri_B_train)
    #plot_temp(a_c, a_h, theta_c, theta_h, theta_ideal, fitted_temp_train, adjust_temp_train, hive_temp_train)
    #plot_temp_A(a_c, a_h, fitted_peri_A_train, adjust_peri_A_train, theta_env_train)
    #plot_temp_B(a_c, a_h, fitted_peri_B_train, adjust_peri_B_train, theta_env_train)
    #plot_temp_S(a_c, a_h, fitted_peri_S_train, adjust_peri_S_train, theta_env_train)

    h_next, theta_ideal_next, reconstructed_input_next = fit_values(data[i_start-window_size*6:i_start+window_size], 
                                          k_W, k_A_W, k_hive, eta_W, eta_A_W, eta_B_W)
    theta_ideal_next += 273.15
    reconstructed_input_next = data[i_start-window_size*6:i_start+window_size]['Temp'].values + reconstructed_input_next
    print(h_next, theta_ideal_next)

    T_0 = [reconstructed_input[-1] + 273.15,theta_env_train[0],theta_env_train[0], theta_env_train[0]]
    print(a_h,a_c,theta_h, theta_c)
    sol = solve_ivp(bee_eq, t_span, T_0, args = [r_c, r_h, t_env_interp_train, h_next, t_solar_interp_train, k_W, k_A_W, k_hive, 
                                                    eta_W, eta_A_W, eta_B_W, theta_ideal_next,
                                                    a_h, a_c, theta_h, theta_c,
                                                    k_c, k_h, C_c, C_h], 
                        t_eval=t_eval, max_step = 0.001, 
                        method= 'LSODA')
    
    adjust_temp_test = sol.y[0,:]
    adjust_peri_A_test = sol.y[1,:]
    adjust_peri_B_test = sol.y[2,:]
    adjust_peri_S_test = sol.y[3,:]

    T_0 = [reconstructed_input[-1] + 273.15,theta_env_train[0],theta_env_train[0], theta_env_train[0]]

    sol = solve_ivp(bee_eq, t_span, T_0, args = [r_c, r_h, t_env_interp_train, h_next, t_solar_interp_train, k_W, k_A_W, k_hive, 
                                                   eta_W, eta_A_W, eta_B_W, theta_ideal_next,
                                                    0, 0, -np.inf, np.inf,
                                                    k_c, k_h, C_c, C_h], 
                        t_eval=t_eval, max_step = 0.001, 
                        method= 'LSODA')
    fitted_temp_test = sol.y[0,:]
    fitted_peri_A_test = sol.y[1,:]
    fitted_peri_B_test = sol.y[2,:]
    fitted_peri_S_test = sol.y[3,:]

    for i in range(24): 
        if i not in selected_points:
            adjust_temp_test[i] = fitted_temp_test[i]
            if fitted_temp_test[i] > theta_c:
                adjust_peri_A_test[i] = fitted_peri_A_test[i]
            if fitted_temp_test[i] < theta_h:
                adjust_peri_B_test[i] = fitted_peri_B_test[i]

    #plot_temp(a_c, a_h, theta_c, theta_h, theta_ideal_next, fitted_temp_test, adjust_temp_test, hive_temp_train)
    #plot_temp_A(a_c, a_h, fitted_peri_A_test, adjust_peri_A_test, theta_env_train)
    #plot_temp_B(a_c, a_h, fitted_peri_B_test, adjust_peri_B_test, theta_env_train)
    #plot_temp_S(a_c, a_h, fitted_peri_S_test, adjust_peri_S_test, theta_env_train)

    h = h_next
    theta_ideal = theta_ideal_next
    reconstructed_input = reconstructed_input_next
    all_temp = np.concatenate((all_temp, np.array([data_train['Date'].values, data_train['Time'].values,
                                                   data_train['DateTime'].values, data_train['Air Temp'].values,         
                                                   data_train['Sol Rad'].values,
                                                   data_train['Temp'].values,
                                                   fitted_temp_train,
                                                   adjust_temp_train,
                                                   fitted_peri_A_train,
                                                   fitted_peri_B_train,
                                                   fitted_peri_S_train,
                                                   fitted_peri_A_test,
                                                   fitted_peri_B_test,
                                                   fitted_peri_S_test,
                                                   adjust_peri_A_train,
                                                   adjust_peri_B_train,
                                                   adjust_peri_S_train,
                                                   adjust_peri_A_test,
                                                   adjust_peri_B_test,
                                                   adjust_peri_S_test,
                                                   fitted_temp_test,
                                                   adjust_temp_test,
                                                   np.array([theta_c]*24),
                                                   np.array([theta_h]*24),
                                                   np.array([h]*24),
                                                   np.array([theta_ideal]*24),
                                                   np.array([h_next]*24),
                                                   np.array([theta_ideal_next]*24)]).reshape(28,-1).T))
all_temp = pd.DataFrame(all_temp, columns = ['Date', 'Time', 'DateTime', 'Air Temp', 'Sol Rad',
                                            'Temp_Actual', 'Forecast_Actual', 'Forecast_Adjust', 
                                            'Top_Forecast_Actual', 'Bottom_Forecast_Actual', 'Side_Forecast_Actual',
                                            'Top_Recons_Actual', 'Bottom_Recons_Actual', 'Side_Recons_Actual',
                                            'Top_Forecast_Adjust', 'Bottom_Forecast_Adjust', 'Side_Forecast_Adjust',
                                            'Top_Recons_Adjust', 'Bottom_Recons_Adjust', 'Side_Recons_Adjust',
                                            'Recons_Actual','Recons_Adjust', 'theta_c', 'theta_h',
                                            'h_forecast','theta_forecast','h_actual','theta_actual'])

all_temp.to_csv('../ABK_codes/C265_profit_control_fcfs_200.csv')