import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import io
import os

def fetch_stock_data(ticker_symbol, start_date, end_date):
    """
    Fetch stock data from Yahoo Finance API
    
    Parameters:
    ticker_symbol (str): The stock ticker symbol (e.g., 'AAPL')
    start_date (datetime): Start date for data retrieval
    end_date (datetime): End date for data retrieval
    
    Returns:
    pandas.DataFrame: DataFrame containing stock data
    """
    try:
        # Convert datetime to string format required by yfinance
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Fetch data from Yahoo Finance
        stock_data = yf.download(ticker_symbol, start=start_str, end=end_str)
        
        # Reset index to make date a column
        stock_data = stock_data.reset_index()
        
        # Add additional calculated columns
        if len(stock_data) > 0:
            # Calculate daily returns
            stock_data['Daily_Return'] = stock_data['Close'].pct_change()
            
            # Calculate moving averages
            stock_data['MA_20'] = stock_data['Close'].rolling(window=20).mean()
            stock_data['MA_50'] = stock_data['Close'].rolling(window=50).mean()
            
            # Calculate volatility (20-day standard deviation)
            stock_data['Volatility_20d'] = stock_data['Daily_Return'].rolling(window=20).std()
            
            # Add ticker symbol as a column
            stock_data['Symbol'] = ticker_symbol
            
        return stock_data
    
    except Exception as e:
        raise Exception(f"Failed to fetch stock data: {str(e)}")

def fetch_market_data(start_date, end_date):
    """
    Fetch market index data (S&P 500) for comparison
    
    Parameters:
    start_date (datetime): Start date for data retrieval
    end_date (datetime): End date for data retrieval
    
    Returns:
    pandas.DataFrame: DataFrame containing market index data
    """
    try:
        # Convert datetime to string format required by yfinance
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Fetch S&P 500 data from Yahoo Finance
        market_data = yf.download('^GSPC', start=start_str, end=end_str)
        
        # Reset index to make date a column
        market_data = market_data.reset_index()
        
        # Add additional calculated columns
        if len(market_data) > 0:
            # Calculate daily returns
            market_data['Daily_Return'] = market_data['Close'].pct_change()
            
            # Calculate moving averages
            market_data['MA_20'] = market_data['Close'].rolling(window=20).mean()
            market_data['MA_50'] = market_data['Close'].rolling(window=50).mean()
            
            # Add index name as a column
            market_data['Index'] = 'S&P 500'
            
        return market_data
    
    except Exception as e:
        raise Exception(f"Failed to fetch market data: {str(e)}")

def parse_uploaded_data(uploaded_file):
    """
    Parse uploaded CSV file containing any type of data
    
    Parameters:
    uploaded_file: The uploaded CSV file object
    
    Returns:
    pandas.DataFrame: DataFrame containing the parsed data
    """
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # Check if it's financial data (has typical financial columns) or generic data
        financial_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        is_financial_data = all(col in df.columns for col in financial_columns)
        
        if is_financial_data:
            # Process as financial data
            # Convert date column to datetime
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Add calculated columns if they don't exist
            if 'Daily_Return' not in df.columns:
                df['Daily_Return'] = df['Close'].pct_change()
            
            if 'MA_20' not in df.columns:
                df['MA_20'] = df['Close'].rolling(window=20).mean()
            
            if 'MA_50' not in df.columns:
                df['MA_50'] = df['Close'].rolling(window=50).mean()
            
            if 'Volatility_20d' not in df.columns:
                df['Volatility_20d'] = df['Daily_Return'].rolling(window=20).std()
                
            # Add symbol column if not present
            if 'Symbol' not in df.columns:
                df['Symbol'] = "CSV_Data"
        else:
            # For non-financial data, add some basic structure
            # Check if there's any date/time column
            date_columns = [col for col in df.columns if any(date_term in col.lower() for date_term in ['date', 'time', 'day', 'year', 'month'])]
            
            # If a date column exists, try to convert it to datetime
            if date_columns:
                try:
                    date_col = date_columns[0]  # Use the first date-like column
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    # Rename to 'Date' for consistency
                    df.rename(columns={date_col: 'Date'}, inplace=True)
                except:
                    # If conversion fails, add a generic date column
                    df['Date'] = pd.date_range(start=pd.Timestamp.now().normalize(), periods=len(df), freq='D')
            else:
                # If no date column exists, add a generic one
                df['Date'] = pd.date_range(start=pd.Timestamp.now().normalize(), periods=len(df), freq='D')
            
            # Add Symbol column for identification
            if 'Symbol' not in df.columns:
                df['Symbol'] = "CSV_Data"
        
        return df
    
    except Exception as e:
        raise Exception(f"Failed to parse uploaded data: {str(e)}")

def load_sample_data():
    """
    Load sample financial data for demonstration purposes
    
    Returns:
    tuple: (financial_data, market_data) both as pandas DataFrames
    """
    try:
        # Define date range for sample data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # Fetch Apple stock data as sample financial data
        financial_data = fetch_stock_data('AAPL', start_date, end_date)
        
        # Fetch S&P 500 data as sample market data
        market_data = fetch_market_data(start_date, end_date)
        
        return financial_data, market_data
    
    except Exception as e:
        raise Exception(f"Failed to load sample data: {str(e)}")

def compute_financial_metrics(data):
    """
    Compute metrics from the data - handles both financial and generic data
    
    Parameters:
    data (pandas.DataFrame): DataFrame containing any type of data
    
    Returns:
    dict: Dictionary of computed metrics
    """
    metrics = {}
    
    if len(data) == 0:
        return metrics
    
    # Check if it's financial data by looking for expected columns
    is_financial_data = all(col in data.columns for col in ['Close', 'High', 'Low', 'Volume'])
    
    # Common metrics for any dataset
    metrics['num_records'] = len(data)
    metrics['num_columns'] = len(data.columns)
    metrics['columns'] = list(data.columns)
    
    # Compute date range if Date column exists
    if 'Date' in data.columns:
        metrics['start_date'] = data['Date'].min().strftime('%Y-%m-%d') if pd.api.types.is_datetime64_any_dtype(data['Date']) else str(data['Date'].min())
        metrics['end_date'] = data['Date'].max().strftime('%Y-%m-%d') if pd.api.types.is_datetime64_any_dtype(data['Date']) else str(data['Date'].max())
        metrics['date_range_days'] = (data['Date'].max() - data['Date'].min()).days if pd.api.types.is_datetime64_any_dtype(data['Date']) else None
    
    if is_financial_data:
        # Financial data specific metrics
        # Price metrics
        metrics['latest_close'] = data['Close'].iloc[-1]
        metrics['price_change_1d'] = data['Close'].iloc[-1] - data['Close'].iloc[-2] if len(data) > 1 else 0
        metrics['price_change_pct_1d'] = (metrics['price_change_1d'] / data['Close'].iloc[-2]) * 100 if len(data) > 1 else 0
        
        # Period changes
        if len(data) >= 5:  # 1 week (5 trading days)
            metrics['price_change_1w'] = data['Close'].iloc[-1] - data['Close'].iloc[-5]
            metrics['price_change_pct_1w'] = (metrics['price_change_1w'] / data['Close'].iloc[-5]) * 100
        
        if len(data) >= 21:  # 1 month (21 trading days)
            metrics['price_change_1m'] = data['Close'].iloc[-1] - data['Close'].iloc[-21]
            metrics['price_change_pct_1m'] = (metrics['price_change_1m'] / data['Close'].iloc[-21]) * 100
        
        # Volatility
        metrics['volatility_20d'] = data['Volatility_20d'].iloc[-1] * 100 if 'Volatility_20d' in data.columns and not pd.isna(data['Volatility_20d'].iloc[-1]) else None
        
        # Trading volume
        metrics['avg_volume_20d'] = data['Volume'].rolling(window=20).mean().iloc[-1] if len(data) >= 20 else None
        metrics['latest_volume'] = data['Volume'].iloc[-1]
        metrics['volume_change_pct'] = ((metrics['latest_volume'] / metrics['avg_volume_20d']) - 1) * 100 if metrics.get('avg_volume_20d') else None
        
        # Price range
        metrics['52w_high'] = data['High'].max() if len(data) >= 252 else data['High'].max()
        metrics['52w_low'] = data['Low'].min() if len(data) >= 252 else data['Low'].min()
        metrics['pct_off_52w_high'] = ((metrics['latest_close'] / metrics['52w_high']) - 1) * 100
        
        # Moving averages
        if 'MA_20' in data.columns and 'MA_50' in data.columns:
            metrics['ma_20'] = data['MA_20'].iloc[-1] if not pd.isna(data['MA_20'].iloc[-1]) else None
            metrics['ma_50'] = data['MA_50'].iloc[-1] if not pd.isna(data['MA_50'].iloc[-1]) else None
            
            if metrics.get('ma_20') and metrics.get('ma_50'):
                metrics['ma_20_50_diff'] = metrics['ma_20'] - metrics['ma_50']
                metrics['ma_20_50_diff_pct'] = (metrics['ma_20_50_diff'] / metrics['ma_50']) * 100
    else:
        # Generic data metrics
        # Find numerical columns
        numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
        metrics['numeric_columns'] = numeric_columns
        
        # Compute basic statistics for each numeric column
        for col in numeric_columns:
            # Basic statistics
            metrics[f'{col}_mean'] = data[col].mean()
            metrics[f'{col}_median'] = data[col].median()
            metrics[f'{col}_min'] = data[col].min()
            metrics[f'{col}_max'] = data[col].max()
            metrics[f'{col}_std'] = data[col].std()
            
            # Check for missing values
            missing_count = data[col].isna().sum()
            metrics[f'{col}_missing'] = missing_count
            metrics[f'{col}_missing_pct'] = (missing_count / len(data)) * 100
            
            # Growth/change metrics (if there are enough records)
            if len(data) > 1:
                first_val = data[col].iloc[0]
                last_val = data[col].iloc[-1]
                if first_val != 0:
                    change_pct = ((last_val - first_val) / abs(first_val)) * 100
                    metrics[f'{col}_change_pct'] = change_pct
        
        # Find categorical columns
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        categorical_columns = [col for col in categorical_columns if col != 'Date']
        metrics['categorical_columns'] = categorical_columns
        
        # For each categorical column, provide value counts for top categories
        for col in categorical_columns[:5]:  # Limit to first 5 categorical columns to avoid overwhelming
            # Get the top 5 most common values
            top_values = data[col].value_counts().head(5).to_dict()
            metrics[f'{col}_top_values'] = top_values
    
    return metrics
