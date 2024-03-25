import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

dataSi = np.loadtxt("C:/Users/MAISON/Desktop/StageM1_LEROY_CASTILLO/Si substrate/Scanning_200GHZ-1320GHZ_Si.txt")
dataair = np.loadtxt("C:/Users/MAISON/Desktop/StageM1_LEROY_CASTILLO/Si substrate/Scanning_200GHZ-1320GHZ_air.txt")

    
def cos(params):
    return np.exp(params[0])*np.cos(2.358*params[1]+params[2])

def fit_data_to_custom_function(x_data, y_data, param_ranges, custom_function):
    """
    Fits data to a custom function using a grid search approach.

    Args:
        x_data: Independent variable data.
        y_data: Dependent variable data.
        param_ranges: A list of tuples, each containing the range for each parameter.
        custom_function: The custom function to fit the data to.

    Returns:
        Optimal parameters for the fit and covariance matrix.
    """
    min_error = float('inf')
    optimal_params = None

    for p0 in np.ndindex(*[len(rng) for rng in param_ranges]):
        try:
            p0_values = [rng[i] for rng, i in zip(param_ranges, p0)]
            popt, _ = curve_fit(custom_function, x_data, y_data, p0=p0_values)
            y_pred = custom_function(x_data, *popt)
            error = np.sum((y_data - y_pred) ** 2)
            if error < min_error:
                min_error = error
                optimal_params = popt
        except:
            continue
    return optimal_params

# Example usage:
# Generate some example data
x_data = dataSi[:,3]
y_data = dataSi[:,2]

# Define ranges for parameters
param_ranges = [(0, 10), (0, 3), (0, 10)]  # Example ranges for parameters

# Fit the data to the custom function
optimal_params = fit_data_to_custom_function(x_data, y_data, param_ranges, cos)

print("Optimal Parameters:", optimal_params)


#plt.figure()
#plt.plot(optimal_params[1],optimal_params[0])
#plt.show()