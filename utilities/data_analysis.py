# -------------------------------------------------------------------------------------------------#

""" Copyright (c) 2024 Asensus Surgical """

""" Code Developed by: Alberto Rota """
""" Supervision: Uriya Levy, Gal Weizman, Stefano Pomati """

# -------------------------------------------------------------------------------------------------#

import math
import numpy as np
from scipy.optimize import curve_fit
from math import log, floor, ceil


def asymptote(y_data):
    # Define the exponential function
    def exponential_function(x, a, b, c):
        return a * np.exp(-b * x) + c

    # Fit the data to the exponential function
    y_data = np.array(y_data)
    x_data = np.arange(len(y_data))
    try:
        params, covariance = curve_fit(exponential_function, x_data, y_data)
    except:
        return None
    # Extract the parameters
    a_fit, b_fit, c_fit = params

    # The horizontal asymptote is given by y = c
    horizontal_asymptote = c_fit

    return horizontal_asymptote


def improvement(y_data):
    # Fit the data to the exponential function
    y_data = np.array(y_data)
    x_data = np.arange(len(y_data))

    # Calculate the improvement
    first_value = y_data[0]
    last_value = y_data[-1]
    improvement = ((last_value - first_value) / first_value) * 100

    return improvement


def millify(n: float) -> str:
    """
    Converts a number into a string with a suffix that indicates its scale (thousands, millions, etc.)

    Parameters:
    n (int, float): The number to be converted.

    Returns:
    str: The converted string.
    """
    millnames = ["", " Th", " M", " B", " T"]

    n = float(n)
    millidx = max(
        0,
        min(
            len(millnames) - 1, int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))
        ),
    )

    return "{:.1f}{}".format(n / 10 ** (3 * millidx), millnames[millidx])


def closest_multiple(number: int, base: int, mode: str = "closest") -> int:
    """
    Find the multiple of `base` closest to `number`, with options for ceiling or floor.

    Parameters:
    - number (int): The target number to find the closest multiple for.
    - base (int): The base multiple to use.
    - mode (str): Determines the method for finding the closest multiple.
      Can be "closest" for the nearest multiple, "inf" for the floor (next lower multiple),
      or "sup" for the ceiling (next higher multiple). Default is "closest".

    Returns:
    int: The closest multiple of `base` to `number` according to the selected mode.
    """
    if mode == "inf":
        return (number // base) * base
    elif mode == "sup":
        return ((number + base - 1) // base) * base
    else:  # mode == "closest"
        lower_multiple = (number // base) * base
        upper_multiple = lower_multiple + base
        return (
            lower_multiple
            if (number - lower_multiple) <= (upper_multiple - number)
            else upper_multiple
        )


def closest_power(number: float, base: int, mode: str = "closest") -> float:
    """
    Find the power of `base` closest to `number`, with options for ceiling or floor.

    Parameters:
    - number (float): The target number to approximate with a power of `base`.
    - base (int): The base for the exponentiation.
    - mode (str): Determines the method for finding the closest power.
      Can be "closest" for the nearest power, "inf" for the floor (next lower power),
      or "sup" for the ceiling (next higher power). Default is "closest".

    Returns:
    float: The closest power of `base` to `number` according to the selected mode.
    """
    if mode == "inf":
        return base ** floor(log(number, base))
    elif mode == "sup":
        return base ** ceil(log(number, base))
    else:  # mode == "closest"
        lower_power = base ** floor(log(number, base))
        upper_power = base ** ceil(log(number, base))
        return (
            lower_power
            if (number - lower_power) <= (upper_power - number)
            else upper_power
        )
