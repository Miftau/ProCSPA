import pandas as pd

def preprocess_input(df):
    # Standardize categorical formatting to match training data
    df['fuel type'] = df['fuel type'].str.strip().str.lower()
    df['manufacturer'] = df['manufacturer'].str.strip().str.title()
    df['model'] = df['model'].str.strip().str.title()
    return df
