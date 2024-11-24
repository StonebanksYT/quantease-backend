from flask import Blueprint, request, jsonify
from app.services.portfolio import fetch_data, optimize_portfolio, generate_efficient_frontier
from app.services.risk import calculate_var, calculate_cvar, calculate_sharpe_ratio, fetch_data
from app.services.options import price_option
from app.services.timeseries import analyze_timeseries
import yfinance as yf

api = Blueprint("api", __name__)

@api.route('/fetch_stock_data', methods=['POST'])
def fetch_stock_data():
    data = request.json
    tickers = data.get("tickers", [])  # List of stock tickers
    start_date = data.get("start_date", "2020-01-01")
    end_date = data.get("end_date", "2023-01-01")
    
    stock_data = {}

    for ticker in tickers:
        stock = yf.Ticker(ticker)
        history = stock.history(start=start_date, end=end_date)
        stock_data[ticker] = history["Close"].tolist()  # Get closing prices

    return jsonify(stock_data)

@api.route('/portfolio', methods=['POST'])
def portfolio():
    data = request.json
    tickers = data['tickers']  
    start_date = data.get('start_date', '2010-01-01')
    end_date = data.get('end_date', '2021-01-01')

    # Fetch stock data
    stock_data = fetch_data(tickers, start_date, end_date)

    # Optimize portfolio
    result = optimize_portfolio(stock_data, tickers)
    return jsonify(result)

@api.route('/efficient_frontier', methods=['POST'])
def efficient_frontier():
    """
    API route to generate the Efficient Frontier for given tickers.
    """
    try:
        data = request.json
        tickers = data.get('tickers')  # Example: ["AAPL", "GOOG", "MSFT"]
        start_date = data.get('start_date', '2010-01-01')
        end_date = data.get('end_date', '2021-01-01')

        # Fetch stock data
        stock_data = fetch_data(tickers, start_date, end_date)

        # Generate efficient frontier
        frontier_data = generate_efficient_frontier(stock_data, tickers)

        return jsonify(frontier_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api.route('/risk_management', methods=['POST'])
def risk_management():
    """
    API route to calculate portfolio risk metrics such as VaR, CVaR, and Sharpe Ratio.
    """
    try:
        data = request.json
        metric = data.get('metric')  # Risk metric ('var', 'cvar', 'sharpe')
        confidence_level = data.get('confidence_level', 0.95)
        tickers = data.get('tickers')  # Example: ["AAPL", "GOOG", "MSFT"]
        start_date = data.get('start_date', '2010-01-01')
        end_date = data.get('end_date', '2021-01-01')

        # Fetch stock data
        stock_data = fetch_data(tickers, start_date, end_date)

        # Calculate daily returns
        returns = stock_data.pct_change().dropna()

        if metric == 'var':
            value = calculate_var(returns, confidence_level)
        elif metric == 'cvar':
            var = calculate_var(returns, confidence_level)
            value = calculate_cvar(returns, var, confidence_level)
        elif metric == 'sharpe':
            value = calculate_sharpe_ratio(returns)

        # Return the calculated risk metric as a scalar value
        return jsonify({"value": value})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api.route('/api/option_pricing', methods=['POST'])
def option_pricing():
    """
    API route to calculate option pricing based on the selected model.
    """
    try:
        data = request.json  # Get the JSON data from the request
        result = price_option(data)  # Call the pricing function with the data
        return jsonify({"call_price": result["call_price"], "put_price": result["put_price"]})  # Return the prices
    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Return an error message if something goes wrong

@api.route('/timeseries', methods=['POST'])
def timeseries():
    data = request.json
    result = analyze_timeseries(data)
    return jsonify(result)
