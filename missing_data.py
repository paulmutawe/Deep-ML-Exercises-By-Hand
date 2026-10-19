import pandas as pd

def solution(df):
    df = df.loc[:, df.isna().mean() <= 0.5]
    df = df.loc[df.isna().mean(axis=1) <= 0.5]
    
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].mean())
        elif not df[column].mode().empty:
            df[column] = df[column].fillna(df[column].mode()[0])
            
    return df.reset_index(drop=True)
