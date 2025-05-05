import json
import os
import numpy as np
import pandas as pd
import re
import nltk
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from financial_data import compute_financial_metrics

# Download required NLTK packages if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

def generate_financial_narrative(financial_data, market_data=None, narrative_type="Quarterly Report", 
                               depth_level=3, target_audience="Investors"):
    """
    Generate a financial narrative based on stock data and market data using a rule-based approach
    
    Parameters:
    financial_data (pandas.DataFrame): DataFrame containing stock data
    market_data (pandas.DataFrame): DataFrame containing market index data
    narrative_type (str): Type of narrative to generate
    depth_level (int): Level of detail in the analysis (1-5)
    target_audience (str): Target audience for the narrative
    
    Returns:
    str: Generated financial narrative
    """
    try:
        # Calculate financial metrics
        metrics = compute_financial_metrics(financial_data)
        
        # Calculate market metrics if market data is available
        market_metrics = None
        if market_data is not None and len(market_data) > 0:
            market_metrics = compute_financial_metrics(market_data)
        
        # Get stock information
        symbol = financial_data['Symbol'].iloc[0] if 'Symbol' in financial_data.columns else "Unknown"
        
        # Determine date range
        start_date = financial_data['Date'].min().strftime('%Y-%m-%d')
        end_date = financial_data['Date'].max().strftime('%Y-%m-%d')
        
        # Initialize sentiment analyzer
        sia = SentimentIntensityAnalyzer()
        
        # Analyze price trend
        first_price = financial_data['Close'].iloc[0]
        last_price = financial_data['Close'].iloc[-1]
        price_change = last_price - first_price
        price_change_pct = (price_change / first_price) * 100
        
        # Determine trend description
        if price_change_pct > 15:
            trend_description = "significant upward"
            sentiment = "very positive"
        elif price_change_pct > 5:
            trend_description = "upward"
            sentiment = "positive"
        elif price_change_pct > -5:
            trend_description = "relatively stable"
            sentiment = "neutral"
        elif price_change_pct > -15:
            trend_description = "downward"
            sentiment = "negative"
        else:
            trend_description = "significant downward"
            sentiment = "very negative"
        
        # Calculate volatility
        volatility = metrics.get('volatility_20d', 0)
        if volatility > 3:
            volatility_desc = "highly volatile"
        elif volatility > 1.5:
            volatility_desc = "moderately volatile"
        else:
            volatility_desc = "showing low volatility"
            
        # Find key dates
        high_date = financial_data.loc[financial_data['High'].idxmax(), 'Date'].strftime('%Y-%m-%d')
        high_price = financial_data['High'].max()
        low_date = financial_data.loc[financial_data['Low'].idxmin(), 'Date'].strftime('%Y-%m-%d')
        low_price = financial_data['Low'].min()
        
        # Get volume information
        avg_volume = metrics.get('avg_volume_20d', 0)
        latest_volume = metrics.get('latest_volume', 0)
        volume_change = metrics.get('volume_change_pct', 0)
        
        # Analyze moving averages
        ma_20 = metrics.get('ma_20', None)
        ma_50 = metrics.get('ma_50', None)
        ma_relation = ""
        if ma_20 and ma_50:
            if ma_20 > ma_50:
                ma_relation = "The 20-day moving average is above the 50-day moving average, suggesting a potential bullish trend."
            else:
                ma_relation = "The 20-day moving average is below the 50-day moving average, indicating a potential bearish trend."
        
        # Market comparison
        market_comparison = ""
        if market_metrics:
            market_first_price = market_data['Close'].iloc[0]
            market_last_price = market_data['Close'].iloc[-1]
            market_change_pct = ((market_last_price - market_first_price) / market_first_price) * 100
            
            if price_change_pct > market_change_pct + 5:
                market_comparison = f"{symbol} has significantly outperformed the broader market (S&P 500) during this period. While the S&P 500 changed by {market_change_pct:.2f}%, {symbol} showed a {price_change_pct:.2f}% change."
            elif price_change_pct > market_change_pct:
                market_comparison = f"{symbol} has slightly outperformed the broader market (S&P 500) during this period. The S&P 500 changed by {market_change_pct:.2f}%, while {symbol} changed by {price_change_pct:.2f}%."
            elif price_change_pct > market_change_pct - 5:
                market_comparison = f"{symbol} has performed similarly to the broader market (S&P 500) during this period. The S&P 500 changed by {market_change_pct:.2f}%, while {symbol} changed by {price_change_pct:.2f}%."
            else:
                market_comparison = f"{symbol} has underperformed compared to the broader market (S&P 500) during this period. While the S&P 500 changed by {market_change_pct:.2f}%, {symbol} showed a {price_change_pct:.2f}% change."
            
        # Generate the narrative based on the narrative type
        if narrative_type == "Quarterly Report":
            title = f"{symbol} Quarterly Financial Report: {start_date} to {end_date}"
        elif narrative_type == "Market Analysis":
            title = f"Market Analysis for {symbol}: {start_date} to {end_date}"
        elif narrative_type == "Stock Performance":
            title = f"{symbol} Stock Performance Analysis: {start_date} to {end_date}"
        elif narrative_type == "Investment Recommendation":
            title = f"Investment Recommendation: {symbol} ({start_date} to {end_date})"
        else:
            title = f"Financial Analysis for {symbol}: {start_date} to {end_date}"
        
        # Define sections based on depth level
        executive_summary = f"""## Executive Summary

During the period from {start_date} to {end_date}, {symbol} has shown a {trend_description} trend with a price change of {price_change_pct:.2f}%. The stock price moved from ${first_price:.2f} to ${last_price:.2f}, while {volatility_desc}. {market_comparison}"""
        
        key_highlights = f"""## Key Highlights

- The highest price of ${high_price:.2f} was reached on {high_date}
- The lowest price of ${low_price:.2f} was recorded on {low_date}
- Average trading volume over the last 20 days: {avg_volume:,.0f} shares
- Latest trading volume: {latest_volume:,.0f} shares ({volume_change:.2f}% compared to the 20-day average)"""
        
        technical_analysis = f"""## Technical Analysis

- Current stock price (as of {end_date}): ${last_price:.2f}
- 20-day Moving Average: ${ma_20:.2f if ma_20 else 'N/A'}
- 50-day Moving Average: ${ma_50:.2f if ma_50 else 'N/A'}
- 20-day Volatility: {volatility:.2f}%

{ma_relation}"""
        
        # Add market comparison section if market data is available
        if market_metrics:
            market_section = f"""## Market Comparison

{market_comparison}

While individual stock performance can vary based on company-specific factors, understanding the broader market context is essential for a comprehensive analysis."""
        else:
            market_section = ""
            
        # Add outlook based on the trend and sentiment
        if sentiment == "very positive":
            outlook = f"Based on the data, {symbol} shows a strong positive trend with significant price appreciation during the analyzed period. The technical indicators suggest continued momentum in the short term."
        elif sentiment == "positive":
            outlook = f"The data indicates that {symbol} has performed positively during the analyzed period. The technical indicators suggest a moderately favorable outlook, though investors should monitor market conditions."
        elif sentiment == "neutral":
            outlook = f"Based on the data, {symbol} has remained relatively stable during the analyzed period. The technical indicators suggest a neutral outlook with potential for movement in either direction depending on future market conditions."
        elif sentiment == "negative":
            outlook = f"The data indicates that {symbol} has shown weakness during the analyzed period. The technical indicators suggest caution, and investors may want to carefully evaluate their position."
        else:  # very negative
            outlook = f"Based on the data, {symbol} has experienced a significant decline during the analyzed period. The technical indicators suggest considerable weakness, and investors should carefully assess the situation."
        
        outlook_section = f"""## Outlook

{outlook}

This analysis is based solely on historical price and volume data and does not incorporate fundamental analysis, news events, or other external factors that may influence future performance."""
        
        # Target audience customization
        audience_note = ""
        if target_audience == "Investors":
            audience_note = "\n\nNote: This report is intended for investors and focuses on performance metrics relevant for investment decisions."
        elif target_audience == "Financial Analysts":
            audience_note = "\n\nNote: This report is tailored for financial analysts and includes detailed technical metrics and market comparisons."
        elif target_audience == "General Public":
            audience_note = "\n\nNote: This report is prepared for a general audience and explains financial concepts in accessible terms."
        elif target_audience == "Board Members":
            audience_note = "\n\nNote: This report is prepared for board members and focuses on high-level performance indicators and strategic implications."
        
        # Combine all sections based on depth level
        sections = [executive_summary, key_highlights]
        
        if depth_level >= 2:
            sections.append(technical_analysis)
            
        if depth_level >= 3 and market_section:
            sections.append(market_section)
            
        if depth_level >= 4:
            sections.append(outlook_section)
            
        # Add disclaimer and audience note
        disclaimer = """## Disclaimer

This report is generated automatically based on historical market data. It does not constitute investment advice. Past performance is not indicative of future results. Always conduct your own research or consult with a financial advisor before making investment decisions."""
        
        sections.append(disclaimer + audience_note)
        
        # Assemble the full narrative
        narrative = f"# {title}\n\n" + "\n\n".join(sections)
        
        return narrative
    
    except Exception as e:
        raise Exception(f"Failed to generate financial narrative: {str(e)}")


def generate_market_overview(market_data, depth_level=3, target_audience="Investors"):
    """
    Generate a market overview narrative based on market index data using a rule-based approach
    
    Parameters:
    market_data (pandas.DataFrame): DataFrame containing market index data
    depth_level (int): Level of detail in the analysis (1-5)
    target_audience (str): Target audience for the narrative
    
    Returns:
    str: Generated market overview
    """
    try:
        # Calculate market metrics
        metrics = compute_financial_metrics(market_data)
        
        # Determine market index information
        index_name = market_data['Index'].iloc[0] if 'Index' in market_data.columns else "S&P 500"
        
        # Determine date range
        start_date = market_data['Date'].min().strftime('%Y-%m-%d')
        end_date = market_data['Date'].max().strftime('%Y-%m-%d')
        
        # Initialize sentiment analyzer
        sia = SentimentIntensityAnalyzer()
        
        # Analyze price trend
        first_price = market_data['Close'].iloc[0]
        last_price = market_data['Close'].iloc[-1]
        price_change = last_price - first_price
        price_change_pct = (price_change / first_price) * 100
        
        # Determine trend description
        if price_change_pct > 10:
            trend_description = "significant upward"
            sentiment = "very positive"
        elif price_change_pct > 3:
            trend_description = "upward"
            sentiment = "positive"
        elif price_change_pct > -3:
            trend_description = "relatively stable"
            sentiment = "neutral"
        elif price_change_pct > -10:
            trend_description = "downward"
            sentiment = "negative"
        else:
            trend_description = "significant downward"
            sentiment = "very negative"
        
        # Calculate volatility
        volatility = metrics.get('volatility_20d', 0)
        if volatility > 2:
            volatility_desc = "highly volatile"
        elif volatility > 1:
            volatility_desc = "moderately volatile"
        else:
            volatility_desc = "showing low volatility"
            
        # Find key dates
        high_date = market_data.loc[market_data['High'].idxmax(), 'Date'].strftime('%Y-%m-%d')
        high_price = market_data['High'].max()
        low_date = market_data.loc[market_data['Low'].idxmin(), 'Date'].strftime('%Y-%m-%d')
        low_price = market_data['Low'].min()
        
        # Get volume information
        avg_volume = metrics.get('avg_volume_20d', 0)
        latest_volume = metrics.get('latest_volume', 0)
        volume_change = metrics.get('volume_change_pct', 0)
        
        # Analyze moving averages
        ma_20 = metrics.get('ma_20', None)
        ma_50 = metrics.get('ma_50', None)
        ma_relation = ""
        if ma_20 and ma_50:
            if ma_20 > ma_50:
                ma_relation = "The 20-day moving average is above the 50-day moving average, suggesting a potential bullish trend in the broader market."
            else:
                ma_relation = "The 20-day moving average is below the 50-day moving average, indicating a potential bearish trend in the broader market."
        
        # Generate title
        title = f"{index_name} Market Overview: {start_date} to {end_date}"
        
        # Define sections based on depth level
        executive_summary = f"""## Market Summary

During the period from {start_date} to {end_date}, the {index_name} has shown a {trend_description} trend with a change of {price_change_pct:.2f}%. The index moved from {first_price:.2f} to {last_price:.2f}, while {volatility_desc}."""
        
        key_highlights = f"""## Key Market Highlights

- The highest index value of {high_price:.2f} was reached on {high_date}
- The lowest index value of {low_price:.2f} was recorded on {low_date}
- Average trading volume over the last 20 days: {avg_volume:,.0f}
- Latest trading volume: {latest_volume:,.0f} ({volume_change:.2f}% compared to the 20-day average)"""
        
        technical_analysis = f"""## Technical Analysis

- Current index value (as of {end_date}): {last_price:.2f}
- 20-day Moving Average: {ma_20:.2f if ma_20 else 'N/A'}
- 50-day Moving Average: {ma_50:.2f if ma_50 else 'N/A'}
- 20-day Volatility: {volatility:.2f}%

{ma_relation}"""
        
        # Add market outlook based on the trend and sentiment
        if sentiment == "very positive":
            outlook = f"Based on the data, the {index_name} shows a strong positive trend with significant appreciation during the analyzed period. The technical indicators suggest continued momentum in the short term."
        elif sentiment == "positive":
            outlook = f"The data indicates that the {index_name} has performed positively during the analyzed period. The technical indicators suggest a moderately favorable outlook, though market conditions should be monitored closely."
        elif sentiment == "neutral":
            outlook = f"Based on the data, the {index_name} has remained relatively stable during the analyzed period. The technical indicators suggest a neutral outlook with potential for movement in either direction depending on future market conditions."
        elif sentiment == "negative":
            outlook = f"The data indicates that the {index_name} has shown weakness during the analyzed period. The technical indicators suggest caution for market participants."
        else:  # very negative
            outlook = f"Based on the data, the {index_name} has experienced a significant decline during the analyzed period. The technical indicators suggest considerable market weakness."
        
        outlook_section = f"""## Market Outlook

{outlook}

This analysis is based solely on historical price and volume data and does not incorporate economic indicators, news events, or other external factors that may influence future market performance."""
        
        # Target audience customization
        audience_note = ""
        if target_audience == "Investors":
            audience_note = "\n\nNote: This market overview is intended for investors and focuses on performance metrics relevant for investment decisions."
        elif target_audience == "Financial Analysts":
            audience_note = "\n\nNote: This market overview is tailored for financial analysts and includes detailed technical metrics and movement analysis."
        elif target_audience == "General Public":
            audience_note = "\n\nNote: This market overview is prepared for a general audience and explains financial concepts in accessible terms."
        elif target_audience == "Board Members":
            audience_note = "\n\nNote: This market overview is prepared for board members and focuses on high-level performance indicators and strategic implications."
        
        # Combine all sections based on depth level
        sections = [executive_summary, key_highlights]
        
        if depth_level >= 2:
            sections.append(technical_analysis)
            
        if depth_level >= 4:
            sections.append(outlook_section)
            
        # Add disclaimer and audience note
        disclaimer = """## Disclaimer

This market overview is generated automatically based on historical market data. It does not constitute investment advice. Past performance is not indicative of future results. Always conduct your own research or consult with a financial advisor before making investment decisions."""
        
        sections.append(disclaimer + audience_note)
        
        # Assemble the full narrative
        narrative = f"# {title}\n\n" + "\n\n".join(sections)
        
        return narrative
    
    except Exception as e:
        raise Exception(f"Failed to generate market overview: {str(e)}")

def format_metrics_for_prompt(metrics):
    """
    Format metrics dictionary into a string for the prompt
    
    Parameters:
    metrics (dict): Dictionary of financial metrics
    
    Returns:
    str: Formatted metrics string
    """
    if not metrics:
        return "No metrics available"
    
    # Format the metrics as a bulleted list
    formatted_str = ""
    for key, value in metrics.items():
        if value is not None:
            # Format numeric values to 2 decimal places
            if isinstance(value, (int, float)):
                if 'pct' in key:
                    formatted_str += f"- {key}: {value:.2f}%\n"
                else:
                    formatted_str += f"- {key}: {value:.2f}\n"
            else:
                formatted_str += f"- {key}: {value}\n"
    
    return formatted_str
