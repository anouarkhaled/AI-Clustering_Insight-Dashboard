"""
Data preprocessing module.
Handles data loading, imputation, encoding, and scaling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_sample_data(name="iris"):
    """
    Load sample datasets (iris or wine).

    Args:
        name: 'iris' or 'wine'

    Returns:
        pd.DataFrame with the dataset
    """
    if name == "iris":
        from sklearn.datasets import load_iris
        X = load_iris(as_frame=True)
        return X.frame
    elif name == "wine":
        from sklearn.datasets import load_wine
        X = load_wine(as_frame=True)
        return X.frame
    else:
        return pd.DataFrame()

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def preprocess(df, numeric_features=None, categorical_features=None, drop_na_threshold=0.5):
    """
    Preprocess data: drop uninformative cols, impute, encode categorical, scale.
    
    Args:
        df: pd.DataFrame
        numeric_features: list of numeric column names (auto-detected if None)
        categorical_features: list of categorical column names (auto-detected if None)
        drop_na_threshold: drop columns with missing rate > (1 - threshold)
    
    Returns:
        X_scaled: pd.DataFrame (scaled, preprocessed)
        numeric_features: list of numeric column names used
        categorical_features: list of categorical column names used
    """
    df = df.copy()
    
    # ================= Drop columns with too many missing values =================
    df = df.loc[:, df.isnull().mean() <= (1 - drop_na_threshold)]
    
    # ================= Identify id-like columns (unique values) =================
    id_like_cols = [c for c in df.columns if df[c].nunique() == len(df)]
    
    # ================= Handle datetime columns =================
    date_cols = df.select_dtypes(include=['datetime', 'datetime64[ns]']).columns.tolist()
    for col in date_cols:
        df[col+'_year'] = df[col].dt.year
        df[col+'_month'] = df[col].dt.month
        df[col+'_weekday'] = df[col].dt.weekday
    df = df.drop(columns=date_cols)  # remove original datetime columns
    
    # ================= Define features for preprocessing =================
    df_features = df.drop(columns=id_like_cols)
    
    if numeric_features is None:
        numeric_features = df_features.select_dtypes(include=[np.number]).columns.tolist()
    if categorical_features is None:
        categorical_features = df_features.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # Remove any id-like or unwanted columns
    numeric_features = [c for c in numeric_features if c not in id_like_cols]
    categorical_features = [c for c in categorical_features if c not in id_like_cols]
    
    # ================= Impute numerical features =================
    X_num = pd.DataFrame()
    if numeric_features:
        imputer_num = SimpleImputer(strategy='median')
        X_num = pd.DataFrame(imputer_num.fit_transform(df[numeric_features]), columns=numeric_features)
    
    # ================= Encode categorical features =================
    X_cat = pd.DataFrame()
    for c in categorical_features:
        if df[c].nunique() <= 20:
            X_cat = pd.concat([X_cat, pd.get_dummies(df[c].astype(str), prefix=c)], axis=1)
        else:
            # high cardinality -> frequency encoding
            freq = df[c].value_counts(normalize=True)
            X_cat[c + '_freq'] = df[c].map(freq).fillna(0)
    
    # ================= Combine numerical and categorical features =================
    X = pd.concat([X_num, X_cat], axis=1)
    
    # ================= Scale =================
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X, numeric_features, categorical_features
