import numpy as np
from scipy.stats import norm
import pandas as pd


def cyclic_pattern_detection(time_series):
    """input pandas series, detect cyclic pattern return True/False
    if True, return frequency, if false, frequency is -1
    """
    # calculate autocorrelation
    auto_corr = [time_series.autocorr(lag=i) for i in range(int(len(time_series)/2))]
    # assume auto_corr value is normal distribution, based on 95% confidence interval, calculate the line for signifence
    critical = norm.ppf(1-0.05/2, loc=np.mean(auto_corr), scale=np.std(auto_corr))
    peak_lag = []
    # select the peak of correlation coefficients
    for i, v in enumerate(auto_corr):
        if v > critical:  # if auto corr value > critical value, consider the correlation is significant
            peak_lag.append(i)
    if len(peak_lag) > 2: # repetitive significant peaks as the rule for cyclic patterns
        lag_diff = pd.Series(peak_lag).diff()  # to calculate period
        period = lag_diff.median()
        return True, period
    else:
        return False, -1