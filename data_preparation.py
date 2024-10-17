import numpy as np
import yfinance as yf
import pandas as pd


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

# Function to replace NaN with the mean of the previous and next values
def interpolation(series):
    for i in range(len(series)):
        if pd.isna(series[i]):
            prev_val = series[i-1] if i > 0 else np.nan
            next_val = series[i+1] if i < len(series)-1 else np.nan
            
            # Calculate the mean, considering edge cases
            if pd.isna(prev_val) and pd.isna(next_val):
                mean_val = np.nan  # No values to average
            elif pd.isna(prev_val):
                mean_val = next_val
            elif pd.isna(next_val):
                mean_val = prev_val
            else:
                mean_val = (prev_val + next_val) / 2
            
            series[i] = mean_val
    return series

d = 4

assets = ["DJI", 'BAMLCC0A0CMTRIV', 'NASDAQ100', 'BAMLHYH0A0HYM2TRIV']

merged_df = yf.download("^DJI", '1994-01-01', '2024-08-02')["Close"]

for asset in assets[1:d]:
    #Read csv file of corresponding asset.
    df = pd.read_csv(asset+'.csv', index_col=0)
    
    #Interpret index as datetime.
    df.index = pd.to_datetime(df.index)
    
    #Replace all non-numeric values by np.nan.
    df = pd.to_numeric(df[asset], errors='coerce').astype('float64')
    
    #Add new df to bigger merged_df.
    merged_df = pd.merge(merged_df, df,            
                     left_index=True, right_index=True, how='outer')

merged_df.columns = assets[:d]

#Remove np.nans by using interpolation.
for asset in merged_df.columns:
    merged_df[asset] = interpolation(merged_df[asset])

#Compute daiy returns.
df = merged_df.pct_change()

increments = {"1w":7, "1m":21, "2m": 42, "3m":63, "6m": 126, 
              "1y": 252, "2y":504}

width = "2m"
C, C_mod, C_eig, C_det = correlation_computation(df, int(increments[width]))
C_collection = pd.concat([C_mod, C_eig, C_det], axis = 1)

merged_df.to_excel("asset_paths.xlsx", index = True)
df.to_excel("returns_asset_paths.xlsx", index = True)
C_collection.to_excel(f"C_collection_{width}.xlsx", index=True)
