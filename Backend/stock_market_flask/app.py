from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import yfinance as yf
from flask_caching import Cache
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:3000"]}})
# Flask-Caching configuration
app.config['CACHE_TYPE'] = 'SimpleCache'
app.config['CACHE_DEFAULT_TIMEOUT'] = 300
cache = Cache(app)


# Fetch SP500 symbols from Wikipedia
def fetch_sp500_symbols():
    try:
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-', regex=False)  # Replace '.' with '-'
        return sp500['Symbol'].unique().tolist()
    except Exception as e:
        print("Error fetching SP500 symbols:", e)
        return []


# Fetch stock data from Yahoo Finance
def fetch_stock_data(symbols, start_date, end_date):
    try:
        df = yf.download(tickers=symbols, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)
        return df
    except Exception as e:
        print("Error fetching stock data:", e)
        return pd.DataFrame()


def fetch_sp500_metadata():
    try:
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-', regex=False)  # Replace '.' with '-'
        sp500 = sp500[['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry']]
        return sp500
    except Exception as e:
        print("Error fetching SP500 metadata:", e)
        return pd.DataFrame()


# Calculate moving average
def calculate_moving_average(data, window):
    return data.rolling(window=window).mean()


# Calculate RSI
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# Calculate MACD
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal_line


def calculate_bollinger_bands(data, window=20):
    sma = data.rolling(window=window).mean()
    std_dev = data.rolling(window=window).std()
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)
    return upper_band, lower_band


@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response


@app.before_request
def log_request():
    print(f"Incoming request: {request.method} {request.url}")
    print(f"Headers: {request.headers}")


@app.route('/api/sp500', methods=['GET'])
@cache.cached()
def get_sp500():
    symbols = fetch_sp500_symbols()
    return jsonify({'symbols': symbols})


@app.route('/api/stock-data', methods=['POST'])
def get_stock_data():
    try:
        data = request.json
        symbols = data.get('symbols', [])
        start_date = data.get('start_date', '2020-01-01')
        end_date = data.get('end_date', '2024-01-01')

        if not symbols:
            return jsonify({'error': 'No symbols provided'}), 400

        # Fetch stock data
        stock_data = fetch_stock_data(symbols, start_date, end_date)

        # Ensure the DataFrame is flattened
        stock_data = stock_data.reset_index()  # Flatten MultiIndex columns

        # Convert to JSON
        return stock_data.to_json(orient='records')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/technical-indicators', methods=['POST'])
def calculate_indicators():
    try:
        data = request.json
        prices = pd.DataFrame(data.get('prices', []))

        if 'close' not in prices.columns:
            return jsonify({'error': 'Missing "close" column in data'}), 400

        prices['MovingAverage'] = calculate_moving_average(prices['close'], window=14)
        prices['RSI'] = calculate_rsi(prices['close'])
        prices['MACD'], prices['SignalLine'] = calculate_macd(prices['close'])

        return prices.to_json(orient='records')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/sp500-metadata', methods=['GET'])
@cache.cached()
def get_sp500_metadata():
    sp500_metadata = fetch_sp500_metadata()
    return sp500_metadata.to_json(orient='records')


@app.route('/api/stock-data-by-sector', methods=['POST'])
def get_stock_data_by_sector():
    try:
        data = request.json
        sector = data.get('sector', '')
        start_date = data.get('start_date', '2020-01-01')
        end_date = data.get('end_date', '2024-01-01')

        if not sector:
            return jsonify({'error': 'Sector not provided'}), 400

        sp500_metadata = fetch_sp500_metadata()
        symbols = sp500_metadata[sp500_metadata['GICS Sector'] == sector]['Symbol'].tolist()

        if not symbols:
            return jsonify({'error': f'No symbols found for sector {sector}'}), 404

        # Fetch stock data
        stock_data = fetch_stock_data(symbols, start_date, end_date)
        stock_data = stock_data.reset_index()
        return stock_data.to_json(orient='records')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/advanced-indicators', methods=['POST'])
def calculate_advanced_indicators():
    try:
        data = request.json
        prices = pd.DataFrame(data.get('prices', []))

        if 'close' not in prices.columns:
            return jsonify({'error': 'Missing "close" column in data'}), 400

        prices['UpperBB'], prices['LowerBB'] = calculate_bollinger_bands(prices['close'])
        return prices.to_json(orient='records')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
