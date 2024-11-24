import math
from scipy.stats import norm

def price_option(data):
    """
    Calculate the price of a European call and put option using the Black-Scholes formula.
    
    Args:
        data (dict): A dictionary containing the input parameters:
            - 'spot_price' (S): Current stock price
            - 'strike_price' (K): Strike price of the option
            - 'time_to_maturity' (T): Time to maturity in years
            - 'risk_free_rate' (r): Risk-free interest rate (annualized)
            - 'volatility' (sigma): Volatility of the underlying stock (annualized)

    Returns:
        dict: A dictionary containing the calculated call and put option prices
            - 'call_price': Price of the call option
            - 'put_price': Price of the put option
    """
    S = data['spot_price']  # Current stock price
    K = data['strike_price']  # Strike price
    T = data['time_to_maturity']  # Time to maturity (in years)
    r = data['risk_free_rate']  # Risk-free interest rate
    sigma = data['volatility']  # Volatility

    # Black-Scholes Model
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    # Calculate call and put prices using the Black-Scholes formula
    call_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    put_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return {"call_price": call_price, "put_price": put_price}
