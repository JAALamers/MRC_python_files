import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from numpy.linalg import eigvals

def generate_correlation_matrix(d,seed):
    """
    Generate a random correlation matrix.
    
    Parameters:
    d (int): The dimension of the correlation matrix.
    seed (int): Value for the seed.
    
    Returns:
    C (array): A d x d correlation matrix.
    """
    np.random.seed(seed)
    
    # Generate a random matrix
    A = np.random.randn(d, d)
    
    # Construct a positive semi-definite matrix
    A = np.dot(A, A.T)
    
    # Normalize the matrix to form a correlation matrix
    D = np.diag(np.sqrt(np.diag(A)))
    D_inv = np.linalg.inv(D)
    C = np.dot(np.dot(D_inv, A), D_inv)
    C = np.round(C, decimals=1)
    
    return C

def principal_submatrix(matrix, rows_to_exclude):
    """
    Construct principal submatrix of 'matrix'.
    
    Parameters:
    matrix (array): d x d matrix.
    rows_to_exclude (list(int)): List containing rows/columns to exclude.
    
    Returns:
    submatrix (array): submatrix of 'matrix'.
    """
    row_mask = np.ones(matrix.shape[0], dtype=bool)
    row_mask[rows_to_exclude] = False

    return matrix[row_mask][:, row_mask]

def ei_d(i,d):
    """
    Generate e^i_d matrix.
    
    Parameters:
    i (int): Index of the matrix.
    d (int): Dimension of the matrix.

    Returns:
    eid (array): dxd matrix containing all zeros 
                    except {i,i}-th element is 1.
    """
    eid = np.zeros((d,d))
    eid[i-1,i-1] = 1
    return eid

def PSD_check(matrix):
    """
    Check if a matrix is PSD using eigenvalues.
    """
    return np.all(np.linalg.eigvalsh(matrix) >= 0) #and np.all(matrix == matrix.T)

def p_plus(matrix, epsilon= 1e-10):
    """
    Project a matrix to the nearest positive definite correlation matrix.

    Parameters:
    matrix (np.ndarray): The input matrix to be projected.
    epsilon (float): A small value to ensure positive definiteness.

    Returns:
    np.ndarray: The projected positive definite correlation matrix.
    """
    # Step 1: Symmetrize the matrix
    matrix = (matrix + matrix.T) / 2
    
    # Step 2: Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    
    # Step 3: Adjust eigenvalues to ensure they are positive
    eigenvalues = np.maximum(eigenvalues, epsilon)
    
    # Step 4: Reconstruct the matrix
    pd_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    # Step 5: Normalize to make it a correlation matrix
    d = np.sqrt(np.diag(pd_matrix))
    pd_matrix = pd_matrix / np.outer(d, d)
    
    # Ensure the diagonal elements are exactly 1
    np.fill_diagonal(pd_matrix, 1)
    
    return pd_matrix


def simulate_MRC_process(C0, kappa, C_bar, A, T, N=1, return_df = False, 
                         use_t = False, t_index = None, seed_value = 30):
    """
    Generate an MRC process.
    
    Parameters:
    C0 (matrix array): C_0
    kappa (matrix array): \\kappa
    C_bar (matrix array): \bar{C}
    A (matrix array): A
    T (int): number of days to compute
    N (int): nr of time points per day
    return_df (binary): do we want it in df format?
    use_t (binary): do we want to use timesteps we put in?
    t_index (array): time points
    seed_value (int): random seed value
    """
    
    np.random.seed(seed_value)
    
    
    #Check weak and strong conditions, and print the result.
    d = C0.shape[0]
    
    M_weak   = kappa @ C_bar + C_bar @ kappa - (d - 2) * A @ A
    M_strong = kappa @ C_bar + C_bar @ kappa - d * A @ A
    
    weak_check = True
    strong_check = True
    
    if not PSD_check(M_weak):
        weak_check = False
    
    if not PSD_check(M_strong):
        strong_check = False
    
    print(eigvals(M_weak))
    print(f'Weak uniqueness check: {weak_check}')
    print(eigvals(M_strong))
    print(f'Strong uniqueness check: {strong_check}')
    

    dt = 1 / N

    # Initialize the process
    Ct, Ct_bar = np.zeros((T*N+1, d, d)), np.zeros((T*N+1, d, d))
    Ct[0], Ct_bar[0] = C0, C0

    # Generate Brownian increments with variance dt
    dB = np.random.normal(0, np.sqrt(dt), (T*N, d, d))
    
    p_plus_activation = np.zeros((T+1))
    
    for t in range(1, T+1):
        
        # Compute the drift term
        drift = kappa @ (C_bar - Ct[t-1])
        drift = drift + drift.T
        
        drift_bar = kappa @ (C_bar - Ct_bar[t-1])
        drift_bar = drift_bar + drift_bar.T
        
        #Compute the diffusion term
        diffusion = np.zeros((d, d))
        for n in range(d):
            
            edn = ei_d(n+1,d)
            sqrt_difn = sqrtm(Ct[t-1] - Ct[t-1] @ edn @ Ct[t-1])
            
            # Check if the result is complex
            if np.iscomplex(sqrt_difn).any():
                p_plus_activation[t] = 0.5
                # If the imaginary part is small, set it to zero
                if np.all(np.abs(sqrt_difn.imag) < 1e-4):
                    sqrt_difn = sqrt_difn.real
                else:
                    raise ValueError("The matrix square root has significant imaginary parts.")
                                
            difn =  A[n,n] * sqrt_difn @ dB[t-1] @ edn
            
            diffusion += (difn + difn.T)
            
        # Update C_t and \bar{C_t} using Euler-Maruyama method      
        Ct[t] = Ct[t-1] + drift * dt + diffusion
        Ct_bar[t] = Ct_bar[t-1] + drift_bar * dt
        
        #p_plus_activation update
        if np.any(np.diag(Ct[t]) != 1) or np.any(eigvals(Ct[t])<0):
            p_plus_activation[t] = 1
        
            #Projection
            Ct[t] = p_plus(Ct[t])
            Ct_bar[t] = p_plus(Ct_bar[t])
    
    if return_df:
        if not use_t:
            t_index = np.arange(0,T+1,1)
            
        reshaped_corr = pd.DataFrame(index = t_index)
        
        eigenvalues_df = pd.DataFrame(index=t_index, columns=[f'λ({i+1})' for i in range(d)])
        determinant_series = pd.DataFrame(index=t_index, columns = ['determinant'])
        
        bar_reshaped_corr = pd.DataFrame(index = t_index)
        
        bar_eigenvalues_df = pd.DataFrame(index=t_index, columns=[f'bar(λ)_{i+1}' for i in range(d)])
        bar_determinant_series = pd.DataFrame(index=t_index, columns = ['bar(determinant)'])
        
        for t in range(0, T+1):
            # Compute eigenvalues and determinant
            eigenvalues = np.linalg.eigvalsh(Ct[t])
            determinant = np.linalg.det(Ct[t])
            
            bar_eigenvalues = np.linalg.eigvalsh(Ct_bar[t])
            bar_determinant = np.linalg.det(Ct_bar[t])
            
            # Store eigenvalues and determinant
            eigenvalues_df.loc[t_index[t], :] = eigenvalues
            determinant_series.loc[t_index[t],:] = determinant
            
            bar_eigenvalues_df.loc[t_index[t], :] = bar_eigenvalues
            bar_determinant_series.loc[t_index[t],:] = bar_determinant
            
            # Reshape correlation matrix into desired format
            for col1 in range(d):
                for col2 in range(col1 + 1, d):
                    key = f'ρ({col1 + 1},{col2 + 1})'
                    bar_key = f'bar(ρ({col1 + 1},{col2 + 1}))'
                    reshaped_corr.loc[t_index, key] = Ct[:, col1, col2]
                    bar_reshaped_corr.loc[t_index, bar_key] = Ct_bar[:, col1, col2]
        return pd.concat([reshaped_corr, eigenvalues_df, determinant_series,
                          bar_reshaped_corr, bar_eigenvalues_df, bar_determinant_series], axis=1), p_plus_activation
        
    
    else:
        return {'C_t': Ct, 'C_t_bar': Ct_bar, 'C_t_eig': (np.linalg.eig(Ct))[0],
            'C_t_det': np.linalg.det(Ct)}