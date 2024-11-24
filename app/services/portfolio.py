import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

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

def calculate_annualized_return(returns):
    """
    Calculate annualized return from daily returns.
    """
    return np.mean(returns) * 252  # 252 trading days in a year

def calculate_annualized_volatility(returns):
    """
    Calculate annualized volatility (risk) from daily returns.
    """
    return np.std(returns) * np.sqrt(252)

def calculate_sharpe_ratio(returns, risk_free_rate=0.03):
    """
    Calculate the Sharpe Ratio for a given set of returns.
    """
    return (calculate_annualized_return(returns) - risk_free_rate) / calculate_annualized_volatility(returns)

def optimize_portfolio(data, tickers):
    """
    Optimize portfolio using Sharpe Ratio.
    Args:
        data (pd.DataFrame): Historical returns of the assets
        tickers (list): List of asset tickers
    Returns:
        dict: Optimized weights, expected return, volatility, and Sharpe ratio
    """
    # Calculate daily returns
    returns = data.pct_change().dropna()

    # Calculate mean returns and covariance matrix
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # Initialize weights (equal distribution)
    num_assets = len(tickers)
    weights = np.ones(num_assets) / num_assets

    # Objective function: Negative Sharpe ratio (we want to maximize Sharpe ratio)
    def negative_sharpe_ratio(weights):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -portfolio_return / portfolio_volatility  # Negative for minimization

    # Constraints: The sum of the weights must be 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

    # Bounds: Each weight between 0 and 1 (no short selling)
    bounds = [(0, 1) for _ in range(num_assets)]

    # Perform optimization to maximize Sharpe ratio
    result = minimize(negative_sharpe_ratio, weights, method='SLSQP', bounds=bounds, constraints=constraints)

    # Get optimized weights
    optimized_weights = result.x
    portfolio_return = np.dot(optimized_weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(optimized_weights.T, np.dot(cov_matrix, optimized_weights)))
    portfolio_sharpe = portfolio_return / portfolio_volatility  # Sharpe ratio

    return {
        "optimized_weights": optimized_weights.tolist(),
        "expected_return": portfolio_return,
        "volatility": portfolio_volatility,
        "sharpe_ratio": portfolio_sharpe,
        "tickers": tickers
    }

def generate_efficient_frontier(data, tickers, num_portfolios=10000, risk_free_rate=0.03):
    """
    Generate the Efficient Frontier for a portfolio.
    Args:
        data (pd.DataFrame): Historical returns of the assets.
        tickers (list): List of stock tickers.
        num_portfolios (int): Number of portfolios to simulate.
        risk_free_rate (float): Risk-free rate for Sharpe ratio calculation.
    Returns:
        dict: Dictionary containing returns, volatilities, and Sharpe ratios.
    """
    # Calculate daily returns
    returns = data.pct_change().dropna()

    # Mean returns and covariance matrix
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # Store results
    results = {
        "Return": [],
        "Volatility": [],
        "Sharpe Ratio": [],
        "Weights": []
    }

    for _ in range(num_portfolios):
        # Generate random portfolio weights
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)  # Ensure weights sum to 1

        # Portfolio return and volatility
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        # Sharpe ratio
        portfolio_sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility

        # Save the results
        results["Return"].append(portfolio_return)
        results["Volatility"].append(portfolio_volatility)
        results["Sharpe Ratio"].append(portfolio_sharpe)
        results["Weights"].append(weights)

    # Return the results as a dictionary (to be used in frontend)
    return results