"""
Cluster profiling module.
Generates cluster profiles and textual summaries.
"""

import pandas as pd
import numpy as np


def profile_clusters(original_df, X_scaled, labels, numeric_features, top_n_features=5):
    """
    Generate cluster profiles (feature means, categorical distributions, summaries).
    
    Args:
        original_df: original pd.DataFrame
        X_scaled: scaled pd.DataFrame
        labels: numpy array of cluster assignments
        numeric_features: list of numeric column names
        top_n_features: number of top features to include in profile
    
    Returns:
        profiles: dict mapping cluster_id -> profile dict with keys:
                  - 'size': number of samples in cluster
                  - 'top_numeric_means': dict of top numeric feature means
                  - 'top_categorical': dict of categorical value distributions
                  - 'summary': text summary
    """
    df = original_df.reset_index(drop=True).copy()
    df['_cluster'] = labels
    profiles = {}
    
    for c in sorted(df['_cluster'].unique()):
        sub = df[df['_cluster'] == c]
        prof = {}
        prof['size'] = len(sub)
        
        # numeric feature means
        if len(numeric_features) > 0:
            means = sub[numeric_features].mean().sort_values(ascending=False)
            prof['top_numeric_means'] = means.head(top_n_features).to_dict()
        else:
            prof['top_numeric_means'] = {}
        
        # top categories frequency
        cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        cat_summary = {}
        for cc in cat_cols:
            cat_summary[cc] = sub[cc].value_counts(normalize=True).head(3).to_dict()
        prof['top_categorical'] = cat_summary
        
        # simple textual summary
        prof['summary'] = generate_text_summary(c, prof)
        profiles[int(c)] = prof
    
    return profiles


def generate_text_summary(cluster_label, prof):
    """
    Generate a simple template-based text summary of a cluster profile.
    
    Args:
        cluster_label: cluster ID
        prof: profile dict with keys 'size', 'top_numeric_means', 'top_categorical'
    
    Returns:
        s: text summary string
    """
    s = f"Cluster {cluster_label}: {prof['size']} items."
    if prof['top_numeric_means']:
        nums = ', '.join([f"{k}: {v:.2f}" for k, v in prof['top_numeric_means'].items()])
        s += " Top numeric features (mean): " + nums + '.'
    if prof['top_categorical']:
        for col, vals in prof['top_categorical'].items():
            if vals:
                top = ', '.join([f"{k} ({v*100:.0f}%)" for k, v in vals.items()])
                s += f" {col} -> {top}."
    return s
