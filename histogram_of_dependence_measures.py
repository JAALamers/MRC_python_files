import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
sns.set_style("whitegrid")

from scipy import stats as st

from data_visualisation import corr_dashboard
from MRC_simulation_2 import simulate_MRC_process


#%%

def tail_dependence_approx(data, q, upper=True):
    """
    Compute tail dependence for increasing quantile levels, approaching extreme tails.
    
    Parameters:
    data (pd.DataFrame or np.ndarray): Input data where each column is a variable.
    q : quantile to compute tail dependence.
    upper (bool): If True, calculate upper tail dependence; else calculate lower tail dependence.
    
    Returns:
    dict: Dictionary containing the tail dependence values for each quantile.
    """
    # Convert data to DataFrame if it's not already
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    
    n_vars = data.shape[1]
    tail_dep_results = np.ones((n_vars,n_vars))  # Store tail dependence matrices for each quantile
    
    for i in range(n_vars):
        for j in range(i, n_vars):
            X = data.iloc[:, i]
            Y = data.iloc[:, j]
            
            if upper:
                # Upper tail dependence
                q_X = X.quantile(q)
                q_Y = Y.quantile(q)
                p_Y = np.mean(Y >= q_Y)
                
                if p_Y > 0:
                    tail_dep_results[i, j] = np.mean((X >= q_X) & (Y >= q_Y)) / p_Y
                    tail_dep_results[j, i] = tail_dep_results[i, j]  # Symmetry
                else:
                    tail_dep_results[i, j] = tail_dep_results[j, i] = np.nan
            else:
                # Lower tail dependence
                q_X = X.quantile(1 - q)
                q_Y = Y.quantile(1 - q)
                p_Y = np.mean(Y <= q_Y)
                
                if p_Y > 0:
                    tail_dep_results[i, j] = np.mean((X <= q_X) & (Y <= q_Y)) / p_Y
                    tail_dep_results[j, i] = tail_dep_results[i, j]  # Symmetry
                else:
                    tail_dep_results[i, j] = tail_dep_results[j, i] = np.nan
    
    return tail_dep_results

#%%
###SIMULATE A LOT OF ASSET PATHS FOR MEASURE ANALYSIS
import itertools

#parameters of period 6
KAPPA_poi  = np.array([0.0054818 , 0.01359227, 0.00866634, 0.06575677])
GAMMA_poi  = np.array([0.00529255, 0.05635028, 0.00319704, 0.06120712])
C_BAR_poi  = np.array([[ 1.        , -0.35294548,  0.89664577,  0.32109373],
                       [-0.35294548,  1.        , -0.33920085,  0.13162917],
                       [ 0.89664577, -0.33920085,  1.        ,  0.30568497],
                       [ 0.32109373,  0.13162917,  0.30568497,  1.        ]])

KAPPA_poi  = np.array([0.005 , 0.015, 0.009, 0.06])
GAMMA_poi  = np.array([0.005, 0.06, 0.003, 0.06])

C_BAR_poi  = np.array([[ 1.        , -0.4,  0.9,  0.3],
                       [-0.4,  1.        , -0.3,  0],
                       [ 0.9, -0.3,  1.        ,  0.5],
                       [ 0.3,  0,  0.5,  1.        ]])


#We choose MU and SIGMA 
MU_poi     = np.zeros(4)
SIGMA_poi  = np.ones(4)

#%%
N = 1
T = 1000
seed_value = 30
plot_MRC_path = False

C_bar = C_BAR_poi
kappa = np.diag(KAPPA_poi)
gamma = np.diag(GAMMA_poi)
C_0 = C_bar

d = len(KAPPA_poi)

#number of estimations tau and spearmann's rho
n = 1000
rhos_MRC = np.zeros((n,d,d))
taus_MRC = np.zeros((n,d,d))
rho_ss_MRC = np.zeros((n,d,d))
utd_MRC = np.zeros((n,d,d))
ltd_MRC = np.zeros((n,d,d))
rhos_noMRC = np.zeros((n,d,d))
taus_noMRC = np.zeros((n,d,d))
rho_ss_noMRC = np.zeros((n,d,d))
utd_noMRC = np.zeros((n,d,d))
ltd_noMRC = np.zeros((n,d,d))

for i in range(n):
    print(i)
    seed_value = i
    #Simulate the MRC process
    C_3 = simulate_MRC_process(C_0, kappa, C_bar, gamma, T, N, 
                                   False, False, None, seed_value)
    C_t = C_3['C_t']
    
    if plot_MRC_path:
        #Simulate MRC again to obtain df structure needed for visualisation purposes.
        df_C_3 = simulate_MRC_process(C_0, kappa, C_bar, gamma, T, N, 
                                          True, False, None, seed_value)
        
        df_C_t = df_C_3[0]
        corr_dashboard(df_C_t, d, False, 'MRC')
    
    X_t_noMRC   = np.zeros((T * N, 4))
    X_t_MRC     = np.zeros((T * N, 4))
    
    for j in range(len(X_t_noMRC)):
            
        mu      = MU_poi
        sigma   = SIGMA_poi
        
        # Ensure mu and sigma have the correct shapes
        mu = np.array(mu).reshape(4)         # Mean vector of length 4
        sigma = np.array(sigma).reshape(4)   # Standard deviation vector of length 4
        
        # Generate standard normal variables
        Z = np.random.normal(0, 1, 4)   # Standard normal variables (mean 0, covariance I)
        
        # Apply the transformations using the same Z for both matrices
        X_t_noMRC[j] = mu + np.diag(sigma) @ np.linalg.cholesky(C_bar) @ Z
        X_t_MRC[j]   = mu + np.diag(sigma) @ np.linalg.cholesky(C_t[j+1]) @ Z
    
    
    df_X_t_noMRC    = pd.DataFrame(data = X_t_noMRC)
    df_X_t_MRC      = pd.DataFrame(data = X_t_MRC)
    
    df_X_t_noMRC.columns    = [1,2,3,4]
    df_X_t_MRC.columns      = [1,2,3,4]
    
    rhos_MRC[i]     = np.array(df_X_t_MRC.corr('pearson'))
    taus_MRC[i]     = np.array(df_X_t_MRC.corr('kendall'))
    rho_ss_MRC[i]   = np.array(df_X_t_MRC.corr('spearman'))
    utd_MRC[i]      = tail_dependence_approx(df_X_t_MRC, 0.99)
    ltd_MRC[i]      = tail_dependence_approx(df_X_t_MRC, 0.99, False)

    rhos_noMRC[i]     = np.array(df_X_t_noMRC.corr('pearson'))
    taus_noMRC[i]     = np.array(df_X_t_noMRC.corr('kendall'))
    rho_ss_noMRC[i]   = np.array(df_X_t_noMRC.corr('spearman'))
    utd_noMRC[i]      = tail_dependence_approx(df_X_t_noMRC, 0.99)
    ltd_noMRC[i]      = tail_dependence_approx(df_X_t_noMRC, 0.99, False)

#%%
C_bar = C_BAR_poi
kappa = np.diag(KAPPA_poi)
gamma = np.diag(GAMMA_poi)
C_0 = C_bar

df_C_3 = simulate_MRC_process(C_0, kappa, C_bar, gamma, T, N, 
                                  True, False, None, seed_value)

df_C_t = df_C_3[0]
corr_dashboard(df_C_t, d, False, 'MRC (low magnitude)')



#%%

import os

# Define the directory where you want to save your plots
save_dir = "hist"

# Create the directory if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


th_tau = 2/np.pi * np.arcsin(C_bar)
th_rho_s = 6/np.pi * np.arcsin(C_bar/2)
th_utd = np.eye(d)
th_ltd = np.eye(d)

n_matrices, n_rows, n_cols = rhos_MRC.shape

# Iterate through all unique index pairs (i, j) such that i < j
for i, j in itertools.combinations(range(n_rows), 2):
    
    # Creating subplots for each histogram (rho, tau, rho_s)
    figrho, axrho = plt.subplots(figsize=(8, 5))
    figtau, axtau = plt.subplots(figsize=(8, 5))
    figrhos, axrhos = plt.subplots(figsize=(8, 5))
    figutd, axutd = plt.subplots(figsize=(8, 5))
    figltd, axltd = plt.subplots(figsize=(8, 5))
    
    print(i, j)
    
    ### rho Histogram ###
    data_MRC = rhos_MRC[:, i, j]
    data_noMRC = rhos_noMRC[:, i, j]
    
    # Plot histogram for rho
    Bins = np.arange(np.min(data_MRC), np.max(data_MRC), 0.005)
    axrho.hist(data_MRC, bins=Bins, alpha=0.7, color='blue', label='MRC $C$')
    axrho.hist(data_noMRC, bins=Bins, alpha=0.7, color='red', label='fixed $C$')
    
    # Calculate and plot vertical lines for the mean
    mean_MRC = np.mean(data_MRC)
    mean_noMRC = np.mean(data_noMRC)
    axrho.axvline(mean_MRC, color='blue', linestyle='dashed', linewidth=1.5, label=f'MRC $C$ mean: {mean_MRC:.3f}')
    axrho.axvline(mean_noMRC, color='red', linestyle='dashed', linewidth=1.5, label=f'fixed $C$ mean: {mean_noMRC:.3f}')
    axrho.axvline(C_bar[i,j], color='black', linestyle='dashed', linewidth=1.5, label=f'Gauss th.: {C_bar[i,j]:.3f}')
    
    # Set title, labels, and grid for rho plot
    axrho.set_title(f'Histogram of estimated $\\rho_{{{i+1},{j+1}}}$ ($T= {T}$, $n ={n_matrices}$)')
    
    axrho.set_ylabel('Frequency')
    axrho.grid(True)
    axrho.legend()
    
    figrho.savefig(f"{save_dir}/rho{i+1}{j+1}_{T}_{n}.png", bbox_inches='tight')
    figrho.savefig(f"{save_dir}/rho{i+1}{j+1}_{T}_{n}.pdf", bbox_inches='tight')
    
    ### tau Histogram ###
    data_MRC = taus_MRC[:, i, j]
    data_noMRC = taus_noMRC[:, i, j]
    
    # Plot histogram for tau
    Bins = np.arange(np.min(data_MRC), np.max(data_MRC), 0.005)
    axtau.hist(data_MRC, bins=Bins, alpha=0.7, color='blue', label='MRC $C$')
    axtau.hist(data_noMRC, bins=Bins, alpha=0.7, color='red', label='fixed $C$')
    
    # Calculate and plot vertical lines for the mean
    mean_MRC = np.mean(data_MRC)
    mean_noMRC = np.mean(data_noMRC)
    axtau.axvline(mean_MRC, color='blue', linestyle='dashed', linewidth=1.5, label=f'MRC $C$ mean: {mean_MRC:.3f}')
    axtau.axvline(mean_noMRC, color='red', linestyle='dashed', linewidth=1.5, label=f'fixed $C$ mean: {mean_noMRC:.3f}')
    axtau.axvline(th_tau[i,j], color='black', linestyle='dashed', linewidth=1.5, label=f'Gauss th.: {th_tau[i,j]:.3f}')
    
    # Set title, labels, and grid for tau plot
    axtau.set_title(f'Histogram of estimated $\\tau_{{{i+1},{j+1}}}$ ($T= {T}$, $n ={n_matrices}$)')
    
    axtau.set_ylabel('Frequency')
    axtau.grid(True)
    axtau.legend()
    
    # Now save the figures in the specified folder
    figtau.savefig(os.path.join(save_dir, f"tau{i+1}{j+1}_{T}_{n}.png"), bbox_inches='tight')
    figtau.savefig(os.path.join(save_dir, f"tau{i+1}{j+1}_{T}_{n}.pdf"), bbox_inches='tight')
    
    ### rho_s Histogram ###
    data_MRC = rho_ss_MRC[:, i, j]
    data_noMRC = rho_ss_noMRC[:, i, j]
    
    # Plot histogram for rho_s
    Bins = np.arange(np.min(data_MRC), np.max(data_MRC), 0.005)
    axrhos.hist(data_MRC, bins=Bins, alpha=0.7, color='blue', label='MRC $C$')
    axrhos.hist(data_noMRC, bins=Bins, alpha=0.7, color='red', label='fixed $C$')
    
    # Calculate and plot vertical lines for the mean
    mean_MRC = np.mean(data_MRC)
    mean_noMRC = np.mean(data_noMRC)
    axrhos.axvline(mean_MRC, color='blue', linestyle='dashed', linewidth=1.5, label=f'MRC $C$ mean: {mean_MRC:.3f}')
    axrhos.axvline(mean_noMRC, color='red', linestyle='dashed', linewidth=1.5, label=f'fixed $C$ mean: {mean_noMRC:.3f}')
    axrhos.axvline(th_rho_s[i,j], color='black', linestyle='dashed', linewidth=1.5, label=f'Gauss th.: {th_rho_s[i,j]:.3f}')
    
    # Set title, labels, and grid for rho_s plot
    axrhos.set_title(f'Histogram of estimated $\\rho^{{S}}_{{{i+1},{j+1}}}$ ($T= {T}$, $n ={n_matrices}$)')
    
    axrhos.set_ylabel('Frequency')
    axrhos.grid(True)
    axrhos.legend()
    
    figrhos.savefig(os.path.join(save_dir, f"rho_s{i+1}{j+1}_{T}_{n}.png"), bbox_inches='tight')
    figrhos.savefig(os.path.join(save_dir, f"rho_s{i+1}{j+1}_{T}_{n}.pdf"), bbox_inches='tight')
    
    ### utd Histogram ###
    data_MRC = utd_MRC[:, i, j]
    data_noMRC = utd_noMRC[:, i, j]
    
    # Plot histogram for rho_s
    Bins = np.arange(np.min(data_MRC), np.max(data_MRC), 0.002)
    axutd.hist(data_MRC, bins=Bins, alpha=0.7, color='blue', label='MRC $C$')
    axutd.hist(data_noMRC, bins=Bins, alpha=0.7, color='red', label='fixed $C$')
    
    # Calculate and plot vertical lines for the mean
    mean_MRC = np.mean(data_MRC)
    mean_noMRC = np.mean(data_noMRC)
    axutd.axvline(mean_MRC, color='blue', linestyle='dashed', linewidth=1.5, label=f'MRC $C$ mean: {mean_MRC:.3f}')
    axutd.axvline(mean_noMRC, color='red', linestyle='dashed', linewidth=1.5, label=f'fixed $C$ mean: {mean_noMRC:.3f}')
    axutd.axvline(th_utd[i,j], color='black', linestyle='dashed', linewidth=1.5, label=f'Gauss th.: {th_utd[i,j]:.3f}')
    
    # Set title, labels, and grid for rho_s plot
    axutd.set_title(f'Histogram of estimated $\\lambda^{{U}}_{{{i+1},{j+1}}}$ ($T= {T}$, $n ={n_matrices}$)')
    
    axutd.set_ylabel('Frequency')
    axutd.grid(True)
    axutd.legend()
    
    ### ltd Histogram ###
    data_MRC = ltd_MRC[:, i, j]
    data_noMRC = ltd_noMRC[:, i, j]
    
    # Plot histogram for rho_s
    Bins = np.arange(np.min(data_MRC), np.max(data_MRC), 0.002)
    axltd.hist(data_MRC, bins=Bins, alpha=0.7, color='blue', label='MRC $C$')
    axltd.hist(data_noMRC, bins=Bins, alpha=0.7, color='red', label='fixed $C$')
    
    # Calculate and plot vertical lines for the mean
    mean_MRC = np.mean(data_MRC)
    mean_noMRC = np.mean(data_noMRC)
    axltd.axvline(mean_MRC, color='blue', linestyle='dashed', linewidth=1.5, label=f'MRC $C$ mean: {mean_MRC:.3f}')
    axltd.axvline(mean_noMRC, color='red', linestyle='dashed', linewidth=1.5, label=f'fixed $C$ mean: {mean_noMRC:.3f}')
    axltd.axvline(th_ltd[i,j], color='black', linestyle='dashed', linewidth=1.5, label=f'Gauss th.: {th_ltd[i,j]:.3f}')
    
    # Set title, labels, and grid for rho_s plot
    axltd.set_title(f'Histogram of estimated $\\lambda^{{L}}_{{{i+1},{j+1}}}$ ($T= {T}$, $n ={n_matrices}$)')
    
    axltd.set_ylabel('Frequency')
    axltd.grid(True)
    axltd.legend()
    
    figutd.savefig(f"{save_dir}/Utd{i+1}{j+1}_{T}_{n}.png", bbox_inches='tight')
    figutd.savefig(f"{save_dir}/Utd{i+1}{j+1}_{T}_{n}.pdf", bbox_inches='tight')

    figltd.savefig(f"{save_dir}/ltd{i+1}{j+1}_{T}_{n}.png", bbox_inches='tight')
    figltd.savefig(f"{save_dir}/ltd{i+1}{j+1}_{T}_{n}.pdf", bbox_inches='tight')
    
    # Show plots
    plt.show()
#%%
### SINGLE RUN

N = 1
T = int(1e6)
seed_value = 30
plot_MRC_path = False

C_bar = C_BAR_poi
kappa = np.diag(KAPPA_poi)
gamma = np.diag(GAMMA_poi)
C_0 = C_bar

d = len(KAPPA_poi)

#Simulate the MRC process
C_3 = simulate_MRC_process(C_0, kappa, C_bar, gamma, T, N, 
                               False, False, None, seed_value)
C_t = C_3['C_t']

if plot_MRC_path:
    #Simulate MRC again to obtain df structure needed for visualisation purposes.
    df_C_3 = simulate_MRC_process(C_0, kappa, C_bar, gamma, T, N, 
                                      True, False, None, seed_value)
    
    df_C_t = df_C_3[0]
    corr_dashboard(df_C_t, d, False, 'MRC')

X_t_noMRC   = np.zeros((T * N, 4))
X_t_MRC     = np.zeros((T * N, 4))

for j in range(len(X_t_noMRC)):
        
    mu      = MU_poi
    sigma   = SIGMA_poi
    
    # Ensure mu and sigma have the correct shapes
    mu = np.array(mu).reshape(4)         # Mean vector of length 4
    sigma = np.array(sigma).reshape(4)   # Standard deviation vector of length 4
    
    # Generate standard normal variables
    Z = np.random.normal(0, 1, 4)   # Standard normal variables (mean 0, covariance I)
    
    # Apply the transformations using the same Z for both matrices
    X_t_noMRC[j] = mu + np.diag(sigma) @ np.linalg.cholesky(C_bar) @ Z
    X_t_MRC[j]   = mu + np.diag(sigma) @ np.linalg.cholesky(C_t[j+1]) @ Z


df_X_t_noMRC    = pd.DataFrame(data = X_t_noMRC)
df_X_t_MRC      = pd.DataFrame(data = X_t_MRC)

df_X_t_noMRC.columns    = [1,2,3,4]
df_X_t_MRC.columns      = [1,2,3,4]
#%%
import scipy
percentile_MRC = pd.DataFrame(scipy.stats.norm.cdf(df_X_t_MRC),
                              columns = df_X_t_MRC.columns)
percentile_noMRC = pd.DataFrame(scipy.stats.norm.cdf(df_X_t_noMRC),
                                columns = df_X_t_noMRC.columns)

sns.histplot(percentile_MRC, x=1, y=3, bins = 100)
plt.show()
sns.histplot(percentile_noMRC, x=1, y=3, bins = 100)
plt.show()

#%%
import copulae as cp
import scipy

u_noMRC = cp.pseudo_obs(df_X_t_noMRC)
u_MRC = cp.pseudo_obs(df_X_t_MRC)

emp_cop_MRC = cp.EmpiricalCopula(u_MRC,smoothing='beta')
emp_cop_noMRC = cp.EmpiricalCopula(u_noMRC,smoothing='beta')

def plot_grid(data, cop_df_emp, title, KDEplot = 1):
    d = len(data.columns)
    
    # Create the grid of subplots
    fig, axes = plt.subplots(d - 1, d - 1, figsize=(16, 16))  # Adjust size as needed
    
    # Initialize a flag to check if any data was plotted
    plotted = [[False] * (d - 1) for _ in range(d - 1)]
    
    for col1 in range(d - 1):
        for col2 in range(col1 + 1, d):
            ax = axes[col2 - 1, col1]  # Place the plot in the correct position
            
            if KDEplot:
                sns.kdeplot(
                    x=data.iloc[:,col1], 
                    y=data.iloc[:,col2],
                    cmap='coolwarm',    # Color map for the KDE
                    fill=True,         # Shade the KDE plot
                    bw_adjust=0.5,      # Bandwidth adjustment
                    thresh=0.05,        # Threshold for the KDE plot
                    ax=ax               # Pass the ax to sns.kdeplot
                )
                plotted[col2 - 1][col1] = True
                
            else:
                sns.histplot(cop_df_emp, x=col1+1, y=col2+1, bins=bins, ax=ax)
                plotted[col2 - 1][col1] = True
                
            # Optionally, set the title for each subplot
            ax.set_title(f'$X_{{{col1+1}}}$ vs $X_{{{col2+1}}}$', fontsize=20)
            ax.set_xlim(0,1)
            ax.set_ylim(0,1)

    # Hide the axes for subplots that were not used
    for i in range(d - 1):
        for j in range(d - 1):
            if not plotted[i][j]:
                axes[i, j].axis('off')  # Hide the axes

    # Adjust layout and add a main title
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    return fig

bins = 100
sample_scale = int(bins / 12) ** 2

cop_df_emp_MRC = emp_cop_MRC.random(10000)
cop_df_emp_noMRC = emp_cop_noMRC.random(10000)

# Call the function for each dataset
fig = plot_grid(cop_df_emp_MRC, cop_df_emp_MRC, 'Pseudo Observations KDE Joint Density Grid (MRC $C$)', 1)
fig.savefig(f"MRC_cop_{T}.png", bbox_inches='tight')
fig.savefig(f"MRC_cop_{T}.pdf", bbox_inches='tight')

fig = plot_grid(cop_df_emp_MRC, cop_df_emp_MRC, 'Pseudo Observations KDE Joint Density Grid (MRC $C$)', 0)
fig.savefig(f"MRC_cop_{T}_dots.png", bbox_inches='tight')
fig.savefig(f"MRC_cop_{T}_dots.pdf", bbox_inches='tight')


fig = plot_grid(cop_df_emp_noMRC, cop_df_emp_noMRC, 'Pseudo Observations 10.000 samples Joint Density Grid (fixed $C$)', 1)
fig.savefig(f"noMRC_cop_{T}.png", bbox_inches='tight')
fig.savefig(f"noMRC_cop_{T}.pdf", bbox_inches='tight')

fig = plot_grid(cop_df_emp_noMRC, cop_df_emp_noMRC, 'Pseudo Observations 10.000 samples Joint Density Grid (fixed $C$)', 0)
fig.savefig(f"noMRC_cop_{T}_dots.png", bbox_inches='tight')
fig.savefig(f"noMRC_cop_{T}_dots.pdf", bbox_inches='tight') 

#%%
###GAUSSIAN COPULA PLOT

gauss_cop = cp.GaussianCopula(dim=d)

print('pct')
gauss_cop.fit(df_X_t_noMRC)

#replace covariance matrix by C_BAR_poi
gauss_cop[:] = C_bar

print(gauss_cop.sigma)

cop_df_gauss = gauss_cop.random(10000)

fig = plot_grid(cop_df_gauss, cop_df_gauss, 'Pseudo Observations KDE Joint Density Grid (Gauss)', 1)
fig.savefig("gauss_cop.png", bbox_inches='tight')
fig.savefig("gauss_cop.pdf", bbox_inches='tight')

fig = plot_grid(cop_df_gauss, cop_df_gauss, 'Pseudo Observations 10.000 samples Joint Density Grid (Gauss)', 0)
fig.savefig("gauss_cop_dots.png", bbox_inches='tight')
fig.savefig("gauss_cop_dots.pdf", bbox_inches='tight')

gmmrets = pd.DataFrame(st.norm.ppf(cop_df_gauss),columns = df_X_t_noMRC.columns)

print('pearson')
print(gmmrets.corr('pearson'))
print()
print('kendall')
print(gmmrets.corr('kendall'))
print(th_tau)
print()
print('spearman')
print(gmmrets.corr('spearman'))
print(th_rho_s)
print("as you can see the derived measure values are near identical to the theoretical values.")

#%%

# Tail dependence paths.

import numpy as np
from scipy.stats import t
from data_visualisation import corr_dashboard_ax1

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
            for j in range(i+1, n):
                
                
                # Get the correlation between variable i and j
                rho = corr_matrix[i, j]
                
                # Compute the argument for the t-distribution CDF
                argument = -np.sqrt((nu + 1) * (1 - rho) / (1 + rho))
                
                # Compute the tail dependence for this pair
                tail_dependence_coefficient = 2 * t.cdf(argument, df=nu+1)
                
                # Fill in the symmetric tail dependence matrix for the current k-th matrix
                tail_dep_array[k, i, j] = tail_dependence_coefficient
                tail_dep_array[k, j, i] = tail_dependence_coefficient
    
    return tail_dep_array


KAPPA_poi  = np.array([0.005 , 0.015, 0.06])
GAMMA_poi  = np.array([0.005, 0.06, 0.06])/3

C_BAR_poi  = np.array([[ 1.        , -0.4,  0.3],
                       [-0.4,  1.        ,  0],
                       [ 0.3,  0,  1.        ]])

N = 1
T = 10000
seed_value = 30
plot_MRC_path = False

C_bar = C_BAR_poi
kappa = np.diag(KAPPA_poi)
gamma = np.diag(GAMMA_poi)
C_0 = C_bar

d = len(kappa)

C_3 = simulate_MRC_process(C_0, kappa, C_bar, gamma, T, N, 
                               False, False, None, seed_value)
C_t = C_3['C_t']


def TD_plot(C, Cbar, TD, TDbar,nu, d, title="MRC process"):
    
    sns.set_style("whitegrid")
    nr_of_unique_elements = int( d * (d-1) / 2 )
    
    # Choosing colors for the plots
    cmap = plt.get_cmap("tab20c")
    norm = plt.Normalize(0, np.max([nr_of_unique_elements, d]))
    colors = cmap(norm(range(np.max([nr_of_unique_elements, d]))))
    
    # Initializing the plot
    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])  # Adjust width ratios
    
    # Create subplots based on gridspec
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[0, 1])
    ax4 = fig.add_subplot(gs[1, 1])
    titles = {
        ax1: 'Path of the off-diagonal elements of the MRC process',
        ax2: f'Paths of computed tail dependence, {title} ($\\nu  = {nu}$)',
        ax3: 'PDF Histograms of MRC',
        ax4: 'PDF Histograms of $\\lambda^L$'}
    
    # Example dummy y-limits
    ylims = {ax1: [-1, 1],
             ax2: [0, 1],
             ax3: [-1, 1],
             ax4: [0, 1]}  # adjust as necessary
    
    color_counter = 0  # Dummy variable for coloring
    
    # Plot correlation paths
    for col1 in range(d):
        for col2 in range(col1+1,d):
            ax1.plot(C[:,col1,col2], label= f'$\\rho_{{{col1 + 1},{col2 + 1}}}$', color=colors[color_counter])
            ax2.plot(TD[:,col1,col2], label= f'$\\lambda^L_{{{col1 + 1},{col2 + 1}}}$', color=colors[color_counter])
            
            ax3.hist(C[:,col1,col2], bins=20, alpha=0.5, density = True,
                     color=colors[color_counter],
                     orientation = 'horizontal')
            
            ax4.hist(TD[:,col1,col2], bins=20, alpha=0.5, density = True,
                     color=colors[color_counter],
                     orientation = 'horizontal')
            
            if col1 == d-2 and col2 == d-1:
                ax3.axhline(Cbar[col1,col2], color=colors[color_counter], linestyle='dashed', linewidth=1.5, 
                            label = '$\\bar{C}$-values')
                ax4.axhline(TDbar[col1,col2], color=colors[color_counter], linestyle='dashed', linewidth=1.5, 
                            label = '$\\lambda^L$ for $\\bar{C}$')
                
            else:
                ax3.axhline(Cbar[col1,col2], color=colors[color_counter], linestyle='dashed', linewidth=1.5)
                ax4.axhline(TDbar[col1,col2], color=colors[color_counter], linestyle='dashed', linewidth=1.5)
                
            
            color_counter += 1
    for ax in [ax1, ax2, ax3, ax4]:
        ax.legend(loc= 'upper left')
        ax.set_title(titles[ax])
        ax.grid(True)
        ax.set_ylim(ylims[ax])
        if ax in [ax1,ax2]:
            ax.set_xlim(0,len(C[:,col1,col2]))
            ax.set_xlabel('time')

        
    plt.tight_layout()
    
    fig.savefig(f"simulation_{title}{nu}.png", bbox_inches='tight')
    fig.savefig(f"simulation_{title}{nu}.pdf", bbox_inches='tight')
    
    plt.show()


for nu in [1]:
    # degrees of freedom for the t-Copula
    
    # Compute the tail dependence array
    lambda_U = tail_dependence_array(C_t, nu)
    lambda_U_bar = tail_dependence_array(C_bar, nu)[0]
    
    TD_plot(C_t,C_bar, lambda_U, lambda_U_bar, nu,d, 
                   'Student-t')
    
#%%
num_points = 1000
rhos = np.ones((num_points,2,2))
rhos[:,0,1] = rhos[:,1,0] = np.linspace(-0.9999,0.9999,num_points)

fig, ax = plt.subplots(figsize=(10,6))

for nu in [1,2,4,16,256]:
    TDs = tail_dependence_array(rhos, nu)
    
    ax.plot(rhos[:,0,1], TDs[:,0,1],label = f'$\\nu={nu}$')
ax.legend()
ax.set_xlabel('correlation $\\rho_{{1,2}}$')
ax.set_ylabel('lower tail dependence coefficient $\\lambda^{{L}}_{{1,2}}$')
ax.set_title('Relation between correlation $\\rho_{{1,2}}$ and lower tail dependence coefficient $\\lambda^{{L}}_{{1,2}}$ of a $t$ copula')

fig.savefig("rho_td.png", bbox_inches='tight')
fig.savefig("rho_td.pdf", bbox_inches='tight')