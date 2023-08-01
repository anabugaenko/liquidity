from scipy.optimize import least_squares
from typing import List, Optional, Tuple
import statsmodels.api as sm
from hurst import compute_Hc
import numpy as np
import pandas as pd

# Helper
def _interpret_hurst(H: float) -> str:
    if not 0 <= H <= 1:
        return "Hurst Exponent not in a valid range [0, 1], series may not be a long memory process"
    if H == 0.5:
        return "Perfect diffusivity: series is a geometric or Brownian random walk"
    if H < 0.5:
        return "Sub-diffusive: series demonstrates anti-persistent behavior"
    if H > 0.5:
        return "Super-diffusive: series demonstrates persistent long-range dependence"
    return "Invalid Hurst Exponent"


class HurstEstimator:
    def __init__(self, ts):
        if ts is None:
            raise ValueError("Time series can't be None")

        # Pre-processing checks
        ts = np.array(ts, dtype=float)
        ts = np.where(ts == None, np.nan, ts)

        if np.any(np.isnan(ts)) or np.any(np.isinf(ts)):
            raise ValueError("Time series contains NaN or Inf values")

        ts = np.where(ts == 0, 1e-10, ts)

        # Store the time series
        self.ts = ts

    def _std_of_sums(self, chunk_size):
        sums = []
        for i in range(0, len(self.ts), chunk_size):  # Iterate over the time series with a step size of chunk_size
            chunk = self.ts[i: i + chunk_size]  # Get the next chunk of size chunk_size
            if len(chunk) == chunk_size:  # If we have a full chunk of size chunk_size
                sums.append(np.sum(chunk))  # Sum up the chunk and add to the list
        return np.std(sums)  # Return the standard deviation of the sums


    def generalized_hurst(self, fitting_method: str ='direct_fit') -> Tuple[float, float, List[float]]:
        if fitting_method not in ['log_log', 'direct_fit']:
            raise ValueError(f"Unknown method: {fitting_method}. Expected 'log_log' or 'direct_fit'.")

        # Default behaviour - Hurst exponent over a range of chunk sizes, starting from 5 and going up to (approximately) half the length of the time series, in steps of 5
        chunk_sizes = range(5, len(self.ts) // 2, 5)

        # Define Hurst function and residuals for optimization
        _hurst_function = lambda N, D, H: D * N ** H
        _residuals = lambda params, N, y_values: y_values - _hurst_function(N, *params)

        # Calculate y_values only for valid chunk sizes
        y_values = []
        valid_chunk_sizes = []
        for chunk_size in chunk_sizes:
            result = self._std_of_sums(chunk_size)
            if result is not None and result != 0:  # Or some other condition for valid results
                y_values.append(result)
                valid_chunk_sizes.append(chunk_size)

        chunk_sizes = valid_chunk_sizes  # Only keep chunk sizes for which _std_of_sums gave a valid result

        if fitting_method == 'direct_fit':
            result = least_squares(_residuals, [0.5, 0.5], loss='soft_l1',
                                   args=(np.array(chunk_sizes), np.array(y_values)))
            D, H = result.x
        else:  # 'log_log'
            log_chunk_sizes = np.log(chunk_sizes)
            log_stds = np.log(y_values)
            X = sm.add_constant(log_chunk_sizes)
            model = sm.RLM(log_stds, X, M=sm.robust.norms.HuberT())
            results = model.fit()
            H = results.params[1]
            D = np.exp(results.params[0])

        # Return the chunk sizes and y_values along with the Hurst exponent and constant D
        return H, D, [list(chunk_sizes), y_values]

    def hurst_from_alpha(self, alpha: float) -> Tuple[float, Optional[float]]:
        H = 1 - alpha / 2
        return H, None

    def rescaled_range(self, kind: str = 'random_walk') -> Tuple[float, float, List[float]]:
        H, c, data = compute_Hc(self.ts, kind=kind)
        return H, c, data

    def estimate(self, method: str = 'generalized_hurst', **kwargs) -> Tuple[float, float, pd.DataFrame, str]:
        if method not in ['rescaled_range', 'hurst_from_alpha', 'generalized_hurst']:
            raise ValueError(f"Unknown method: {method}")

        if method == 'rescaled_range' or method == 'generalized_hurst':
            H, const, data = getattr(self, method)(**kwargs)
        else:
            H, const = getattr(self, method)(**kwargs)
            data = None

        interpretation = _interpret_hurst(H)
        return H, const, data, interpretation
