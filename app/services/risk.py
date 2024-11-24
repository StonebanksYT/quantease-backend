import numpy as np
import pandas as pd
import yfinance as yf

def fetch_data(tickers, start_date="2010-01-01", end_date="2021-01-01"):
    """
    Fetch historical stock data from Yahoo Finance.
    Args:
        tickers (list): List of stock tickers (e.g., ['AAPL', 'GOOG', 'MSFT'])
        start_date (str): Start date for historical data (default: '2010-01-01')
        end_date (str): End date for historical data (default: '2021-01-01')
    Returns:
        pd.DataFrame: DataFrame of historical adjusted close prices for the tickers
    """
    data = yf.download(tickers, start=start_date, end=end_date)["Adj Close"]
    return data

def calculate_var(returns, confidence_level=0.95):
    """
    Calculate the Value at Risk (VaR) using the historical simulation method.
    Args:
        returns (pd.Series or pd.DataFrame): A series or DataFrame of daily returns.
        confidence_level (float): The confidence level for VaR (default 95%).
    Returns:
        float: The Value at Risk (VaR) at the specified confidence level.
    """
    if isinstance(returns, pd.DataFrame):
        # If returns is a DataFrame, we assume the user wants to calculate VaR on the first column
        returns = returns.iloc[:, 0]  # Use the first column for VaR calculation (you can change this as needed)

    # Now, returns should be a Series, so we can sort it directly
    sorted_returns = returns.sort_values()

    # Calculate the index for the desired quantile (VaR)
    var_index = int((1 - confidence_level) * len(sorted_returns))

    # VaR is the return at the quantile index
    var = sorted_returns.iloc[var_index]  # VaR is the quantile at the given confidence level
    return var

def calculate_cvar(returns, var, confidence_level=0.95):
    """
    Calculate the Conditional Value at Risk (CVaR), also known as Expected Shortfall.
    Args:
        returns (pd.Series): A series of daily returns.
        var (float): The Value at Risk at the specified confidence level.
        confidence_level (float): The confidence level for CVaR (default 95%).
    Returns:
        float: The Conditional Value at Risk (CVaR) at the specified confidence level.
    """
    # CVaR is the average of returns that are less than or equal to VaR
    cvar = returns[returns <= var].mean()  # This will return a scalar (float)
    return float(cvar)  # Ensure we return a scalar (float), which is JSON serializable

def calculate_sharpe_ratio(returns, risk_free_rate=0.03):
    """
    Calculate the Sharpe Ratio for a given set of returns.
    Args:
        returns (pd.Series): A series of daily returns.
        risk_free_rate (float): The risk-free rate for Sharpe ratio calculation (default 3%).
    Returns:
        float: The Sharpe ratio.
    """
    # Sharpe ratio = (mean return - risk-free rate) / standard deviation
    sharpe_ratio = (returns.mean() - risk_free_rate) / returns.std()  # This will return a scalar (float)
    return float(sharpe_ratio)  # Ensure we return a scalar (float), which is JSON serializable
