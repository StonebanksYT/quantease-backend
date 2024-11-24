from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize

app = Flask(__name__)
api = Api(app)

# Download stock data from Yahoo Finance
def download_stock_data(ticker, start_date=None, end_date=None):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date) if start_date and end_date else stock.history(period="1y")
        if data.empty:
            raise ValueError(f"No data found for ticker: {ticker}")
        return data
    except Exception as e:
        raise ValueError(f"Error fetching data for ticker {ticker}: {str(e)}")

# Portfolio Optimization
def construct_optimal_portfolio(tickers):
    stock_data = {}
    for ticker in tickers:
        try:
            stock_data[ticker] = download_stock_data(ticker)['Close']
        except ValueError as e:
            return {"error": str(e)}, 400

    df = pd.DataFrame(stock_data)
    returns = df.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    risk_free_rate = 0.02 / 252  # 2% annualized risk-free rate converted to daily

    # Objective function to minimize: negative Sharpe Ratio
    def neg_sharpe_ratio(weights):
        portfolio_return = np.sum(mean_returns * weights) * 252  # Annualize return
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)  # Annualize volatility
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        return -sharpe_ratio  # Minimize the negative Sharpe Ratio

    # Constraints and bounds
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # The sum of weights must equal 1
    bounds = tuple((0, 1) for _ in range(len(tickers)))  # No short selling allowed
    initial_weights = len(tickers) * [1.0 / len(tickers)]  # Equal initial weights

    # Optimization to find the best weights
    result = minimize(neg_sharpe_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    if not result.success:
        return {"error": "Optimization failed"}, 400

    optimal_weights = result.x
    portfolio_return = np.sum(mean_returns * optimal_weights) * 252  # Annualized return
    portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights))) * np.sqrt(252)  # Annualized volatility
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility  # Sharpe Ratio

    # Return the optimized portfolio information
    return {
        "tickers": tickers,
        "weights": optimal_weights.tolist(),
        "expected_return": portfolio_return,
        "volatility": portfolio_volatility,
        "sharpe_ratio": sharpe_ratio
    }

# API Resources

class Portfolio(Resource):
    def post(self):
        tickers = request.json.get('tickers')
        if not tickers or not isinstance(tickers, list):
            return {"error": "List of tickers is required"}, 400

        # Call the portfolio optimization function
        portfolio = construct_optimal_portfolio(tickers)
        return portfolio, 200

# Add resources to API
api.add_resource(Portfolio, '/portfolio/construct')

if __name__ == '__main__':
    app.run(debug=True)
