import numpy as np
import pandas as pd
from scipy.fft import fft
import pywt
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_regression

class SeismicFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, wavelet='db4', levels=3, window_size=5, n_features=30):
        self.wavelet = wavelet
        self.levels = levels
        self.window_size = window_size
        self.n_features = n_features
        self.feature_selector = None
        
    def _extract_wavelet_features(self, signal):
        # Calculate maximum possible decomposition level
        max_level = pywt.dwt_max_level(len(signal), self.wavelet)
        levels = min(self.levels, max_level)
        
        coeffs = pywt.wavedec(signal, self.wavelet, level=levels)
        features = []
        for coeff in coeffs:
            # Handle precision issues in moment calculations
            if len(coeff) > 1:
                stats = [
                    np.mean(coeff), np.std(coeff), np.max(coeff), np.min(coeff),
                    skew(coeff, nan_policy='omit'), 
                    kurtosis(coeff, nan_policy='omit'), 
                    np.median(coeff)
                ]
            else:
                # For single-value coefficients, use simpler stats
                stats = [coeff[0], 0, coeff[0], coeff[0], 0, 0, coeff[0]]
            features.extend(stats)
        return np.array(features)
    
    def _extract_frequency_features(self, signal):
        fft_vals = np.abs(fft(signal))
        freqs = np.fft.fftfreq(len(signal))
        features = [
            np.mean(fft_vals), np.std(fft_vals), np.max(fft_vals),
            freqs[np.argmax(fft_vals)], np.median(fft_vals),
            np.sum(fft_vals > np.mean(fft_vals))
        ]
        return np.array(features)
    
    def _extract_statistical_features(self, signal):
        rolling_mean = pd.Series(signal).rolling(window=self.window_size, min_periods=1).mean().fillna(0)
        rolling_std = pd.Series(signal).rolling(window=self.window_size, min_periods=1).std().fillna(0)
        peaks, _ = find_peaks(signal)
        return np.array([
            np.mean(signal), np.std(signal), np.max(signal), np.min(signal),
            skew(signal), kurtosis(signal), len(peaks),
            np.nanmean(rolling_mean), np.nanmean(rolling_std),
            np.percentile(signal, 25), np.percentile(signal, 75)
        ])
    
    def _extract_time_domain_features(self, signal):
        zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
        energy = np.sum(signal**2)
        entropy = -np.sum(np.where(signal != 0, signal * np.log(signal), 0))
        return np.array([zero_crossings, energy, entropy])
    
    def transform(self, X, y=None):
        features = []
        for signal in X:
            wavelet_features = self._extract_wavelet_features(signal)
            freq_features = self._extract_frequency_features(signal)
            stat_features = self._extract_statistical_features(signal)
            time_features = self._extract_time_domain_features(signal)
            combined_features = np.concatenate([
                wavelet_features, freq_features, stat_features, time_features
            ])
            features.append(combined_features)
        
        features = np.array(features)
        if self.feature_selector:
            features = self.feature_selector.transform(features)
        return features
    
    def fit(self, X, y=None):
        if y is not None:
            # Extract all features first
            all_features = self.transform(X)
            # Select top n features based on correlation with target
            self.feature_selector = SelectKBest(f_regression, k=self.n_features)
            self.feature_selector.fit(all_features, y)
        return self
