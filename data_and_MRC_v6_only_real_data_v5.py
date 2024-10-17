import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import datetime
from scipy.optimize import lsq_linear
from numpy.linalg import eigvals
from scipy.optimize import minimize
import time
import seaborn as sns
from scipy import stats as st

#CHANGE POINT
import ruptures as rpt
from ruptures.costs import CostNormal

#OWN FUNCTIONS
from data_visualisation import kAC_dashboard, setup_axis, plot_with_rectangles
from data_visualisation import corr_dashboard, corr_dashboard_ax1, TD_plot
from MRC_simulation_2 import simulate_MRC_process, PSD_check

#COPULAS
from copulae import StudentCopula
from scipy.stats import t

#%%
def correlation_computation(df, width):
    # Compute the rolling correlation with a window size of 'width'
    rolling_corr = df.rolling(width).corr()
    
    # Initialize DataFrames for reshaped correlation, eigenvalues, and determinant
    reshaped_corr = pd.DataFrame()
    
    num_columns = len(df.columns)
    
    eigenvalues_df = pd.DataFrame(index=df.index, columns=[f'λ({i+1})' for i in range(num_columns)])
    determinant_series = pd.DataFrame(index=df.index, columns = ['determinant'])
    
    # Iterate over the rolling correlation matrices
    for i in range(width, len(df)):
        window_corr_matrix = rolling_corr.loc[df.index[i]]
        
        if window_corr_matrix.isnull().values.any():
            # If the correlation matrix contains NaNs, skip this window
            continue
        
        # Compute eigenvalues and determinant
        eigenvalues = np.linalg.eigvalsh(np.array(window_corr_matrix))
        determinant = np.linalg.det(window_corr_matrix)

        # Store eigenvalues and determinant
        eigenvalues_df.loc[df.index[i], :] = eigenvalues
        determinant_series.loc[df.index[i], :] = determinant
        
        # Reshape correlation matrix into desired format
        for col1 in range(len(df.columns)):
            for col2 in range(col1 + 1, len(df.columns)):
                key = f'ρ({col1 + 1},{col2 + 1})'
                reshaped_corr.loc[df.index[i], key] = window_corr_matrix.iloc[col1, col2]
    
    first_valid_index = reshaped_corr.notnull().all(axis=1).idxmax()

    # Slice the DataFrame from the first valid index to the end
    reshaped_corr = reshaped_corr.loc[first_valid_index:]
    eigenvalues_df = eigenvalues_df.loc[first_valid_index:]
    determinant_series = determinant_series.loc[first_valid_index:]
    
    return rolling_corr, reshaped_corr, eigenvalues_df, determinant_series

d = 4
def elbow_plot(data, algo, window):
    # Store the costs for different numbers of change points
    costs = []
    n_bkps_list = range(1, 16)  # testing 1 to 10 change points
    
    for n_bkps in n_bkps_list:
        # Run the segmentation
        result = algo.fit(data).predict(n_bkps=n_bkps)
        
        # Compute the cost (sum of squared errors)
        cost = algo.cost.sum_of_costs(result)
        costs.append(cost)
        
    # Plot the elbow plot
    plt.figure(figsize=(8, 5))
    plt.plot(n_bkps_list, costs, marker='o')
    plt.title(f"Elbow Plot for Change Point Detection (sliding window {window})")
    plt.xlabel("Number of Change Points")
    plt.ylabel("Cost (Sum of Squared Errors)")
    
    plt.savefig(f"Elbow_plot_{window}.png", bbox_inches='tight')
    plt.savefig(f"Elbow_plot_{window}.pdf", bbox_inches='tight')
    
    plt.show()
    
def Get_Change_points(data, number_of_bkps, window, 
                      plot_elbow, plot_disc_curve):
    #model = {'l2','normal'}
    model = 'normal'
    
    algo = rpt.Window(width=window*2, model = model, jump=1, 
                      params={"add_small_diag": 1e-10}).fit(data)
    
    my_bkps = algo.predict(n_bkps=number_of_bkps)
    
    if plot_elbow:
        elbow_plot(data, algo, window)
    
    if plot_disc_curve:
        # model = rpt.costs.CostL2()  # using l2 norm for the cost function
        #model =  CostMeanVar()
        model = CostNormal(add_small_diag=1e-10)
        model.fit(data)
        
        # Initialize a list to store discrepancies
        discrepancies = []
        
        # Compute the discrepancy for each position of the sliding window
        for i in range(window, len(data) - window):
            cost = model.error(i - window, i + window)
            cost_left = model.error(i - window, i)  # cost of the left window
            cost_right = model.error(i, i + window)  # cost of the right window
            discrepancy = cost - cost_left - cost_right
            discrepancies.append(discrepancy)
        
        # Pad the discrepancy list to align with the original time series
        discrepancies = [0] * window + discrepancies + [0] * window
    
        # Plot the discrepancy curve
        plt.figure(figsize=(10, 6))
        plt.plot(data.index, discrepancies)
        plt.title(f"Discrepancy curve (sliding window {window})")
        plt.xlabel("Time")
        plt.ylabel("Discrepancy")
        
        plt.savefig(f"discrepancy_curve_{window}.png", bbox_inches='tight')
        plt.savefig(f"discrepancy_curve_{window}.pdf", bbox_inches='tight')
        
        plt.show()
    
    my_bkps[-1] = my_bkps[-1]-1

    bkps_periods = [[data.index[0], 
                  data.index[my_bkps[0]],
                  'Per. 1']]
    
    for i in range(0, len(my_bkps)-1):
        bkps_periods.append([data.index[my_bkps[i]], 
                      data.index[my_bkps[i+1]],
                      f'Per. {i+2}'])
        
    return my_bkps, bkps_periods

def ols_jacobi_kappa(data, dt, rho_bar):
    # define variables
    Nsteps = len(data)
    rho_s = data[:Nsteps - 1]  
    rho_t = data[1:Nsteps]
    
    # feature engineering to fit the theoretical model
    y = (rho_t - rho_s) / np.sqrt(1 - rho_s**2)
    z = ((rho_bar - rho_s) * dt / np.sqrt(1 - rho_s**2)).reshape(-1, 1)  

    # fit the model
    model = LinearRegression(fit_intercept=False)
    model.fit(z, y)

    # calculate the predicted values (y_hat), residuals, and the parameters
    y_hat = model.predict(z)
    residuals = y - y_hat
    beta1 = model.coef_

    # get the parameter of interest for CIR
    k = beta1
    gamma = np.std(residuals) / np.sqrt(dt)
    
    return k, gamma

def ols_jacobi(data, dt):

    # define variables
    Nsteps = len(data)
    rho_s = data[:Nsteps - 1]  
    rho_t = data[1:Nsteps]
    
    # model initialization
    model = LinearRegression()

    # feature engineering to fit the theoretical model
    y = (rho_t - rho_s) / np.sqrt( 1 - rho_s**2 )
    z1 = dt / np.sqrt( 1 - rho_s**2 )
    z2 = dt * rho_s / np.sqrt( 1 - rho_s**2 )
    Z = np.column_stack((z1, z2))

    # fit the model
    model = LinearRegression(fit_intercept=False)
    model.fit(Z, y)

    # calculate the predicted values (y_hat), residuals and the parameters
    y_hat = model.predict(Z)
    residuals = y - y_hat
    beta1 = model.coef_[0]        
    beta2 = model.coef_[1]

    # get the parameter of interest for CIR
    k = -beta2
    rho_bar = beta1/k
    gamma = np.std(residuals)/np.sqrt(dt)
    
    return k, rho_bar, gamma

def pair_splitter(pair_estimates,d):
    K = np.zeros((int(d*(d-1)/2),d))
    b = pair_estimates
    counter=0
    for i in range(d):
        for j in range(i+1,d):
            K[counter,i], K[counter,j] = 1,1
            counter += 1
    
    bounds = (0, np.inf)
    
    # Solve the system of equations
    result = lsq_linear(K, b, bounds=bounds)
    
    if result.success:
        k = result.x
    else:
        raise ValueError("Optimization did not converge")
    
    return k

def OLS_MRC(rhos, estimate_C_bar = True, C_bar = np.eye(d)):
    """
    Fit MRC parameters to derived correlation paths.
    
    Parameters:
    rhos (df): Dataframe containing correlation paths.

    Returns:
    kappa_estimate (matrix):    parameter MRC.
    C_bar_estimate (matrix):    parameter MRC.
    A_estimate (matrix):        parameter MRC.
    """
    kappa_estimates = np.zeros(len(rhos.columns))
    gamma_estimates = np.zeros(len(rhos.columns))
    C_bar_estimates = np.eye(d)

    counter = 0 #Dummy variable
    
    for col1 in range(d):
        for col2 in range(col1 + 1, d):
            
            if estimate_C_bar:
                estimates = ols_jacobi(np.array(rhos[f'ρ({col1+1},{col2+1})']), 1)
                C_bar_estimates[col1,col2]  = estimates[1]
                
            elif not estimate_C_bar:
                estimates = ols_jacobi_kappa(np.array(rhos[f'ρ({col1+1},{col2+1})']), 
                                             1, C_bar[col1,col2]) 
            
            kappa_estimates[counter]    = estimates[0]
            gamma_estimates[counter]    = estimates[-1]
    
            counter += 1
    
    kappa_estimate = pair_splitter(kappa_estimates, d)

    A_estimate = np.sqrt(pair_splitter(gamma_estimates**2, d)) 
    
    if estimate_C_bar:
        C_bar_estimate = np.array( C_bar_estimates + C_bar_estimates.T- np.eye(d) )
    elif not estimate_C_bar:
        C_bar_estimate = C_bar
        
    return kappa_estimate, C_bar_estimate, A_estimate

def second_moment_MRC(index, kappa, C_bar, A):
    i,j,k,l = int(index[0]-1), int(index[1]-1), int(index[2]-1), int(index[3]-1)
    
    kappa_sum = kappa[i,i] + kappa[j,j] + kappa[k,k] + kappa[l,l]
    k_ijkl = kappa_sum + A[i,i]**2 * ((i==k)+(i==l)) + A[j,j]**2 * ((j==k)+(j==l))

    covariance =  kappa_sum * C_bar[i,j] * C_bar[k,l]
    covariance += A[i,i]**2 * ((i==k) * C_bar[j,l] + (i==l) * C_bar[j,k])
    covariance += A[j,j]**2 * ((j==k) * C_bar[i,l] + (j==l) * C_bar[i,k])
    covariance = (covariance) / k_ijkl
    
    return covariance

def m_hat(df, result):
    return np.sum(df[f'ρ({result[0]},{result[1]})'] * df[f'ρ({result[2]},{result[3]})']) / (len(df))

def penalty_function(kappa, C_bar, A, pen_val):
    M_weak = kappa @ C_bar + C_bar @ kappa - (d) * A @ A
    eigenvalues = eigvals(M_weak)
    penalty = np.sum(np.minimum(eigenvalues, 0)**2)  # Penalize negative eigenvalues
    return penalty * pen_val  # Scale the penalty to be large

def moment_parameter_fit(df, kappa_estimate, C_bar_estimate, A_estimate, 
                         pen_val, plotting=False):
    
    d = len(kappa_estimate)
    
    # Generate all possible (i, j, k, l) combinations
    results = [
        (i, j, k, l)
        for i in range(1, d+1)
        for j in range(i+1, d+1)
        for k in range(i, d+1)
        for l in range(k+1, d+1)
        if i < k or (i == k and j <= l)
    ]
    
    m_hats = np.zeros((d, d, d, d))
    
    for result in results:
        m_hats[result[0]-1, result[1]-1, result[2]-1, result[3]-1] = m_hat(df, result)
    
    def objective_function(params):
        kappa_op = params[:d]
        A_op = params[d:]
        
        total_sum = 0
        for result in results:
            expected = second_moment_MRC(result, np.diag(kappa_op), 
                                         C_bar_estimate, np.diag(A_op))
            target = m_hats[result[0]-1, result[1]-1, result[2]-1, result[3]-1]
            total_sum += (expected - target) ** 2
        return total_sum + penalty_function(np.diag(kappa_op), 
                                            C_bar_estimate, 
                                            np.diag(A_op), pen_val)
    
    initial_guess = np.concatenate((kappa_estimate, A_estimate))
    
    function_values = []
    parameter_values = []
    
    def callback(params):
        function_values.append(objective_function(params))
        parameter_values.append(params.copy())

    # Define the constraints: kappa > 0 and A > 0
    constraints = []
    for i in range(2*d):
        constraints.append({'type': 'ineq', 'fun': lambda params, i=i: params[i]})
    
    options = {'maxiter': 100}
    result_min = minimize(objective_function, initial_guess, callback=callback, 
                          constraints=constraints, options=options)
    
    parameter_values = np.array(parameter_values)
    # print(objective_function(initial_guess))
    # print(function_values)
    
    if plotting:
        plt.figure(figsize=(10, 5))
        plt.plot(function_values, label='Objective Function Value')
        plt.xlabel('Iteration')
        plt.ylabel('Function Value')
        plt.title('Objective Function Value vs. Iteration')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        plt.show()
        
    return result_min.x[:d], C_bar_estimate, result_min.x[d:], result_min.nit

#%%
""" 
Read prepared data, and derive bkps
"""
d = 4

increments = {"1w":7, "1m":21, "3m":63, "6m": 126, 
              "1y": 252, "2y":504}

width = "2m"

df = pd.read_excel("data_prep/asset_paths.xlsx", index_col=0)
df_pct = pd.read_excel("data_prep/returns_asset_paths.xlsx", index_col=0)
C_stat = pd.read_excel(f"data_prep/C_collection_{width}.xlsx", index_col=0)

RHO = C_stat[[col for col in C_stat.columns if col.startswith('ρ(')]]



window_width_bkps = '1y'
num_bkps = 11

plot_elbow = False
plot_disc_curve = False

bkps, bkps_periods = Get_Change_points(RHO, 
                                       num_bkps, 
                                       increments[window_width_bkps],
                                       plot_elbow, plot_disc_curve)
#Show bkps and correlation paths.
corr_dashboard_ax1(C_stat, d, False, 
               f'Correlation matrix ({width}), (bkps window = {increments[window_width_bkps]})', 
               bkps_periods)
#%%
"""
Fit the t_copula values of real data. And find the periodically estimates.
"""
t_cop = StudentCopula(d)

history     = 3 #in months

C_bar_time  = True
OLS         = True

execution_times = []

KAPPAS_period, A_period = np.zeros((len(bkps_periods),d)), np.zeros((len(bkps_periods),d))
C_BARS_period = np.zeros((len(bkps_periods),d,d))
MU_period, SIGMA_period = np.zeros((len(bkps_periods),d)), np.zeros((len(bkps_periods),d))
NU_period = np.zeros(len(bkps_periods))
C_period_t_cop = np.zeros((len(bkps_periods),d,d))

for i in range(len(bkps_periods)):
    start_date      = bkps_periods[i][0]
    end_date        = bkps_periods[i][1]
    
    # Start timer
    start_time = time.time()
    
    RHO_period          = RHO.loc[start_date:end_date]
    C_bar_period        = np.array(df_pct[start_date:end_date].corr())
    
    if execution_times == [] or OLS:
        kappa_estimate, C_bar_estimate_OLS, A_estimate = OLS_MRC(RHO_period)
        
    else:
        kappa_estimate, A_estimate = kappa_MOMENT, A_MOMENT
        
    kappa_MOMENT, C_bar_MOMENT, A_MOMENT, niter = moment_parameter_fit(RHO_period,
                                                                kappa_estimate, 
                                                                C_bar_period,
                                                                A_estimate,
                                                                1e8,
                                                                False) #<-plot
    #ADMINISTRATION PART
    KAPPAS_period[i]    = kappa_MOMENT
    A_period[i]    = A_MOMENT
    C_BARS_period[i]    = C_bar_MOMENT
    
    returns_period = np.array(df_pct[start_date:end_date])
    
    MU_period[i] = np.median(returns_period, axis=0)
    SIGMA_period[i] = np.array([st.median_abs_deviation(returns_period[:, i], scale='normal') for i in range(returns_period.shape[1])])

    kappa = np.diag(kappa_MOMENT)
    A = np.diag(A_MOMENT)
    C_bar = C_bar_period
    M_weak   = kappa @ C_bar + C_bar @ kappa - (d - 2) * A @ A
    M_strong = kappa @ C_bar + C_bar @ kappa - d * A @ A
    
    weak_check = True
    strong_check = True
    
    if not PSD_check(M_weak):
        weak_check = False
    
    if not PSD_check(M_strong):
        strong_check = False
    
    # COPULA PART
    t_cop.fit(returns_period)
    t_cop_param = t_cop.params
    
    NU_period[i] = t_cop_param[0]
    C_period_t_cop[i] = t_cop[:]
    
    # End timer and calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    execution_times.append(elapsed_time)
    
    print(f"{bkps_periods[i][2]}: {niter}, {elapsed_time:.3f} s, Weak: {np.min(eigvals(M_weak)):.1e}, Strong: {np.min(eigvals(M_strong)):.1e}")
print(f"Total computation time: {np.sum(execution_times):.3f} s")

parameters_period_df    = pd.DataFrame()

for i in range(0,d):
    parameters_period_df[f'κ({i+1})'] = KAPPAS_period[:,i]
    parameters_period_df[f'γ({i+1})'] = A_period[:,i]
    
for i in [(i, j) for i in range(1, d+1) for j in range(i+1, d+1)]:
    parameters_period_df[f'bar_ρ({i[0]},{i[1]})'] = C_BARS_period[:,i[0]-1,i[1]-1]
#%%
"""
Compute the tail dependence values of t-copula, and plot it.
"""

formatted_nu = " & ".join([f"{nu:.1f}" for nu in NU_period])
print(f"NU per period: {formatted_nu}")

N = 1
T = 10000
seed_value = 30

#Generate barplots.
def tail_dependence_array(corr_matrix_array, nu):
    if len(corr_matrix_array.shape) == 3:
        # Get the number of correlation matrices and size of each matrix
        m, n, _ = corr_matrix_array.shape
    elif len(corr_matrix_array.shape) == 2:
        m = 1
        n, _ = corr_matrix_array.shape
    # Initialize an array to store the tail dependence matrices
    tail_dep_array = np.zeros((m, n, n))
    
    # Iterate over each correlation matrix
    for k in range(m):
        # Get the correlation matrix for the current index
        if len(corr_matrix_array.shape) == 3:
            corr_matrix = corr_matrix_array[k]
        elif len(corr_matrix_array.shape) == 2:
            corr_matrix = corr_matrix_array
        
        # Iterate over the pairs of variables (i, j)
        for i in range(n):
            for j in range(i + 1, n):
                # Get the correlation between variable i and j
                rho = corr_matrix[i, j]
                
                # Compute the argument for the t-distribution CDF
                argument = -np.sqrt((nu + 1) * (1 - rho) / (1 + rho))
                
                # Compute the tail dependence for this pair
                tail_dependence_coefficient = 2 * t.cdf(argument, df=nu + 1)
                
                # Fill in the symmetric tail dependence matrix for the current k-th matrix
                tail_dep_array[k, i, j] = tail_dependence_coefficient
                tail_dep_array[k, j, i] = tail_dependence_coefficient
    
    return tail_dep_array

# Assuming KAPPAS_period, A_period, C_BARS_period, NU_period are arrays
period_counter = 1
for kappa, gamma, C_bar, nu in zip(KAPPAS_period, A_period, C_BARS_period, NU_period):
    
    kappa_diag = np.diag(kappa)
    gamma_diag = np.diag(gamma)
    
    # Simulate the MRC process here with the new values
    C_3 = simulate_MRC_process(C_bar, kappa_diag, C_bar, gamma_diag, T, N, 
                               False, False, None, seed_value)
    C_t = C_3['C_t']
    
    # Compute the tail dependence array
    lambda_U = tail_dependence_array(C_t, nu)
    lambda_U_bar = tail_dependence_array(C_bar, nu)[0]
    
    # Generate the plot for the current set of parameters
    TD_plot(C_t, C_bar, lambda_U, lambda_U_bar, np.round(nu, 1), len(kappa), 
            f'Student-t{period_counter}_')

    period_counter += 1

#%%
""" 
Estimate \\kappa and A daily
(Can be skipped if df is already generated)
"""

execution_times = [0]

KAPPAS, A = np.zeros((len(RHO),d)), np.zeros((len(RHO),d))
C_BARS = np.zeros((len(RHO),d,d))

counter = 0

RHO_index = RHO.index
RHO_index_offset = RHO_index - pd.DateOffset(months=history)

for j in range(1,len(RHO)):
    start_time = time.time()
    
    end_date = RHO_index[j]
    
    #Find the corresponding period we are in.
    for i, (start, end, period_name) in enumerate(bkps_periods):
        if start <= end_date < end:
            period_index = i
    
    #Use corresponding parameters for given period.
    start_date = max(bkps_periods[period_index][0], RHO_index_offset[j])  
    
    kappa_estimate      = KAPPAS_period[period_index]
    C_bar_estimate      = C_BARS_period[period_index]
    A_estimate          = A_period[period_index]
    
    RHO_period_est = RHO.loc[start_date:end_date]
    
    kappa_MOMENT, C_bar_MOMENT, A_MOMENT, niter = moment_parameter_fit(RHO_period_est,
                                                                    kappa_estimate, 
                                                                    C_bar_estimate,
                                                                    A_estimate,
                                                                    1e6,
                                                                    False)
    
    KAPPAS[j]   = kappa_MOMENT
    A[j]   = A_MOMENT
    C_BARS[j]   = C_bar_MOMENT
        
    # End timer and calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    execution_times.append(elapsed_time)
    est_time_remaining = np.sum(execution_times)/len(execution_times) * (len(RHO) - j)
    ETA = datetime.datetime.fromtimestamp(end_time + est_time_remaining)
        
    print(f"period {period_index+1}, {j}: {niter}, {elapsed_time:.3f} s, {ETA.strftime('%H:%M')}")
print(f"Total computation time: {np.sum(execution_times):.3f} s")

parameters_df = pd.DataFrame()
for i in range(0,d):
    parameters_df[f'κ({i+1})'] = KAPPAS[:,i]
    parameters_df[f'γ({i+1})'] = A[:,i]

for i in [(i, j) for i in range(1, d+1) for j in range(i+1, d+1)]:
    parameters_df[f'bar_ρ({i[0]},{i[1]})'] = C_BARS[:,i[0]-1,i[1]-1]

# Add execution_times to parameters_df
parameters_df['execution_times'] = pd.Series([0]+execution_times)

parameters_df.index = RHO_index

parameters_df.to_excel(f"param_est_paths_w{width}_h{history}.xlsx", index = True)
#%%
""" 
Read data of previously generated df, previous step can be skipped, 
if the df is already generated 
"""

parameters_df = pd.read_excel(f"param_est_paths_w{width}_h{history}.xlsx", 
                              index_col=0)

kAC_dashboard(parameters_df, width, history, parameters_period_df, log = False, 
              years = bkps_periods, ncol = 9)

#%%
"""
Artifical data part!
Generate artificial dataset as shown in thesis
"""

np.random.seed(30)

width = '6m'

T = int(365*2)  # Number of days per period #7800
N = 1           # Number of time steps per day
d = 4           #dimension
d_prime = 4     #if lower dimension prefered

n_bkps = 6
C_bars = np.zeros((n_bkps+1,d,d))
C_bars[0] = np.array(np.round(df_pct[0:1000].corr(),1))
C_bars[1] = np.array(np.round(df_pct[1000:2000].corr(),1))
C_bars[2] = np.array(np.round(df_pct[1000:2000].corr(),1))
C_bars[3] = np.array(np.round(df_pct[1000:2000].corr(),1))
C_bars[4] = np.array(np.round(df_pct[4000:5000].corr(),1))
C_bars[5] = np.array(np.round(df_pct[6000:7000].corr(),1))
C_bars[6] = np.array(np.round(df_pct[6000:7000].corr(),1))

kappa_arrays = np.zeros((n_bkps+1,d))
kappa_arrays[0] = np.array([0.1,0.1,0.1,0.1])*0.5
kappa_arrays[1] = np.array([0.1,0.1,0.1,0.1])*0.5
kappa_arrays[2] = np.array([0.1,0.1,0.1,0.1])*2
kappa_arrays[3] = np.array([0.1,0.1,0.1,0.1])*2
kappa_arrays[4] = np.array([0.1,0.1,0.1,0.1])*1
kappa_arrays[5] = np.array([0.1,0.1,0.1,0.1])*1
kappa_arrays[6] = np.array([0.1,0.1,0.1,0.1])*1/4

A_arrays = np.zeros((n_bkps+1,d))
A_arrays[0] = np.array([0.004, 0.006, 0.003, 0.007])*5
A_arrays[1] = np.array([0.004, 0.006, 0.003, 0.007])*5
A_arrays[2] = np.array([0.004, 0.006, 0.003, 0.007])*5
A_arrays[3] = np.array([0.004, 0.006, 0.003, 0.007])*10
A_arrays[4] = np.array([0.004, 0.006, 0.003, 0.007])*10
A_arrays[5] = np.array([0.004, 0.006, 0.003, 0.007])*2
A_arrays[6] = np.array([0.004, 0.006, 0.003, 0.007])*2/2

A = np.diag(A_arrays[0][:d_prime])
kappa = np.diag(kappa_arrays[0][:d_prime])

C_1, reflection = simulate_MRC_process(C_bars[0], kappa, C_bars[0], A, 
                                       T, N, True, False)
for i in range(1,n_bkps+1):
    C_0 = np.eye(d)
    for col1 in range(d_prime):
        for col2 in range(col1+1,d_prime):
            C_0[col1,col2] =  C_0[col2, col1] = C_1[f'ρ({col1+1},{col2+1})'].iloc[-1]
    
    A = np.diag(A_arrays[i][:d_prime])
    kappa = np.diag(kappa_arrays[i][:d_prime])
    
    C_2, reflection = simulate_MRC_process(C_0, kappa, C_bars[i], A, 
                                           T, N, True, False)

    C_1 = pd.concat([C_1,C_2]).reset_index(drop=True)

# Define the start date
start_date = pd.to_datetime('2000-01-01')

# Create the date range by adding the array values as timedeltas (days) 
# to the start date
dates = start_date + pd.to_timedelta(C_1.index, unit='D')
C_1.index = pd.to_datetime(dates)
C_stat = C_1

increments = {"1w":7, "1m":21, "3m":63, "6m": 126, 
              "1y": 252, "2y":504}

RHO = C_stat[[col for col in C_stat.columns if col.startswith('ρ(')]]

num_bkps = n_bkps
window_width_bkps = '1y'

bkps, bkps_periods = Get_Change_points(RHO, 
                                       num_bkps, 
                                       increments[window_width_bkps],
                                       plot_elbow = False,
                                       plot_disc_curve = True)

corr_dashboard(C_stat, d, False, 'MRC process with change points', bkps_periods)
#%%
""" 
Estimate \\kappa and A periodically and daily
(Can be skipped if df is already generated)
"""

history     = 6    #months

C_bar_time  = True
OLS         = True

execution_times = []


KAPPAS_period, A_period = np.zeros((len(bkps_periods),d)), np.zeros((len(bkps_periods),d))
C_BARS_period = C_bars

for i in range(len(bkps_periods)):
    start_date      = bkps_periods[i][0]
    end_date        = bkps_periods[i][1]
    end_date_offset = bkps_periods[i][1] 
    # Start timer
    start_time = time.time()
    
    RHO_period          = RHO.loc[start_date:end_date]
    C_bar_period        = C_bars[i]
     
    if execution_times == [] or OLS:
        kappa_estimate, C_bar_estimate_OLS, A_estimate = OLS_MRC(RHO_period)
        
    else:
        kappa_estimate, A_estimate = kappa_MOMENT, A_MOMENT
        
    kappa_MOMENT, C_bar_MOMENT, A_MOMENT, niter = moment_parameter_fit(RHO_period,
                                                                kappa_estimate, 
                                                                C_bar_period,
                                                                A_estimate,
                                                                1e6,
                                                                False) #<- plot

    KAPPAS_period[i] = kappa_MOMENT
    A_period[i] = A_MOMENT
    C_BARS_period[i] = C_bar_MOMENT
    
    kappa = np.diag(kappa_MOMENT)
    A = np.diag(A_MOMENT)
    C_bar = C_bar_period
    M_weak   = kappa @ C_bar + C_bar @ kappa - (d - 2) * A @ A
    M_strong = kappa @ C_bar + C_bar @ kappa - d * A @ A
    
    weak_check = True
    strong_check = True
    
    if not PSD_check(M_weak):
        weak_check = False
    
    if not PSD_check(M_strong):
        strong_check = False
    
    
    # End timer and calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    execution_times.append(elapsed_time)
    
    print(f"{bkps_periods[i][2]}: {niter}, {elapsed_time:.3f} s, Weak: {np.min(eigvals(M_weak)):.1e}, Strong: {np.min(eigvals(M_strong)):.1e}")
print(f"Total computation time: {np.sum(execution_times):.3f} s")

parameters_period_df    = pd.DataFrame()
actual_period_df        = pd.DataFrame()

for i in range(0,d):
    parameters_period_df[f'κ({i+1})'] = KAPPAS_period[:,i]
    parameters_period_df[f'γ({i+1})'] = A_period[:,i]
    actual_period_df[f'κ({i+1})'] = kappa_arrays[:,i]
    actual_period_df[f'γ({i+1})'] = A_arrays[:,i]
    

for i in [(i, j) for i in range(1, d+1) for j in range(i+1, d+1)]:
    parameters_period_df[f'bar_ρ({i[0]},{i[1]})'] = C_BARS_period[:,i[0]-1,i[1]-1]



execution_times = [0]

KAPPAS, A = np.zeros((len(RHO),d)), np.zeros((len(RHO),d))
C_BARS = np.zeros((len(RHO),d,d))

counter = 0

RHO_index = RHO.index
RHO_index_offset = RHO_index - pd.DateOffset(months=history)

for j in range(1,len(RHO)):
    start_time = time.time()
    
    end_date = RHO_index[j]
    
    #Find the corresponding period we are in.
    for i, (start, end, period_name) in enumerate(bkps_periods):
        if start <= end_date < end:
            period_index = i
    
    #Use corresponding parameters for given period.
    start_date = max(bkps_periods[period_index][0], RHO_index_offset[j])  
    
    kappa_estimate      = KAPPAS_period[period_index]
    C_bar_estimate      = C_BARS_period[period_index]
    A_estimate          = A_period[period_index]
    
    RHO_period_est = RHO.loc[start_date:end_date]
    
    kappa_MOMENT, C_bar_MOMENT, A_MOMENT, niter = moment_parameter_fit(RHO_period_est,
                                                                    kappa_estimate, 
                                                                    C_bar_estimate,
                                                                    A_estimate,
                                                                    1e6,
                                                                    False) #<- plot
    
    KAPPAS[j]   = kappa_MOMENT
    A[j]   = A_MOMENT
    C_BARS[j]   = C_bar_MOMENT
        
    # End timer and calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    execution_times.append(elapsed_time)
    est_time_remaining = np.sum(execution_times)/len(execution_times) * (len(RHO) - j)
    ETA = datetime.datetime.fromtimestamp(end_time + est_time_remaining)
        
    print(f"period {period_index+1}, {j}: {niter}, {elapsed_time:.3f} s, {ETA.strftime('%H:%M')}")
print(f"Total computation time: {np.sum(execution_times):.3f} s")


parameters_df = pd.DataFrame()
for i in range(0,d):
    parameters_df[f'κ({i+1})'] = KAPPAS[:,i]
    parameters_df[f'γ({i+1})'] = A[:,i]

for i in [(i, j) for i in range(1, d+1) for j in range(i+1, d+1)]:
    parameters_df[f'bar_ρ({i[0]},{i[1]})'] = C_BARS[:,i[0]-1,i[1]-1]


# Add execution_times to parameters_df
parameters_df['execution_times'] = pd.Series([0]+execution_times)

parameters_df.index = RHO_index

parameters_df.to_excel(f"param_est_paths_h{history}_art.xlsx", index = True)
#%%
history = 6
width = 'artificial'
parameters_df = pd.read_excel(f"param_est_paths_h{history}_art.xlsx", index_col=0)

kAC_dashboard(parameters_df, width,
                      history,
                      parameters_period_df,
                      log = False, 
                      years = bkps_periods,
                      ncol = 10,
                      actual_df = actual_period_df)