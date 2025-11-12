import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
class PowerRPMExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts numeric power/rpm from a 'max_power' string like '74bhp@5000rpm'
    and replaces the column with the ratio (power/rpm).
    """
    def __init__(self, power_col='max_power'):
        self.power_col = power_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        def safe_extract(pattern, text):
            try:
                result = re.findall(pattern, str(text))
                return float(result[0]) if result else np.nan
            except Exception:
                return np.nan

        df[self.power_col] = df[self.power_col].apply(
            lambda x: safe_extract(r'\d+\.?\d*(?=bhp)', x) / safe_extract(r'\d+\.?\d*(?=rpm)', x)
            if pd.notnull(x) else np.nan
        )

        df[self.power_col] = pd.to_numeric(df[self.power_col], errors='coerce')

        return df


class TorqueRPMExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts numeric torque/rpm from a 'max_torque' string like '120Nm@4000rpm'
    and replaces the column with the ratio (torque/rpm).
    """
    def __init__(self, torque_col='max_torque'):
        self.torque_col = torque_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        def safe_extract(pattern, text):
            try:
                result = re.findall(pattern, str(text))
                return float(result[0]) if result else np.nan
            except Exception:
                return np.nan

        df[self.torque_col] = df[self.torque_col].apply(
            lambda x: safe_extract(r'\d+\.?\d*(?=Nm)', x) / safe_extract(r'\d+\.?\d*(?=rpm)', x)
            if pd.notnull(x) else np.nan
        )

        # Force numeric conversion (handles "object" dtype)
        df[self.torque_col] = pd.to_numeric(df[self.torque_col], errors='coerce')

        return df



class IndexBasedEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, col, target='is_claim', threshold=1.2, add_target_encoding=True):
        self.col = col
        self.target = target
        self.threshold = threshold
        self.add_target_encoding = add_target_encoding
        self.over_indexed_categories_ = None
        self.global_mean_ = None
        self.claim_rate_map_ = None

    def fit(self, X, y=None):
        df = X.copy()
        if y is not None:
            df[self.target] = y
        
        self.global_mean_ = df[self.target].mean()
        index_df = (
            df.groupby(self.col)[self.target]
            .mean()
            .reset_index()
            .rename(columns={self.target: 'claim_rate'})
        )
        index_df['index'] = index_df['claim_rate'] / self.global_mean_
        
        self.over_indexed_categories_ = index_df.loc[index_df['index'] >= self.threshold, self.col].tolist()
        self.claim_rate_map_ = index_df.set_index(self.col)['claim_rate'].to_dict()
        return self

    def transform(self, X):
        df = X.copy()
        df[self.col + '_grouped'] = df[self.col].apply(
            lambda x: x if x in self.over_indexed_categories_ else 'Other'
        )
        one_hot = pd.get_dummies(df[self.col + '_grouped'], prefix=self.col)
        if self.add_target_encoding:
            df[self.col + '_te'] = df[self.col].map(self.claim_rate_map_)
            df[self.col + '_te'].fillna(self.global_mean_, inplace=True)
            return pd.concat([one_hot, df[[self.col + '_te']]], axis=1)
        else:
            return one_hot
        
    def get_feature_names_out(self, input_features=None):
        """Return output feature names after encoding."""
        # Get one-hot column names (known from fit)
        one_hot_features = [f"{self.col}_{cat}" for cat in self.over_indexed_categories_] + [f"{self.col}_Other"]

        if self.add_target_encoding:
            one_hot_features.append(f"{self.col}_te")

        return np.array(one_hot_features)

