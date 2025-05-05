import json
import re
import os
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sentiment import SentimentIntensityAnalyzer

# Download required NLTK packages if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

def check_narrative_consistency(narrative, financial_data):
    """
    Check the probabilistic consistency of a generated financial narrative
    against the actual financial data
    
    Parameters:
    narrative (str): The generated financial narrative
    financial_data (pandas.DataFrame): The financial data used to generate the narrative
    
    Returns:
    tuple: (consistency_report, consistency_score)
        consistency_report (dict): Detailed report of consistency checks
        consistency_score (float): Overall consistency score between 0 and 1
    """
    try:
        # Extract factual claims from the narrative
        factual_claims = extract_factual_claims(narrative)
        
        # Verify each claim against the financial data
        consistency_checks = []
        for claim in factual_claims:
            verification = verify_claim_against_data(claim, financial_data)
            consistency_checks.append(verification)
        
        # Calculate overall consistency score
        if consistency_checks:
            consistency_score = sum(check['consistency_score'] for check in consistency_checks) / len(consistency_checks)
        else:
            consistency_score = 0.0
        
        # Create a detailed report
        consistency_report = {
            "overall_score": consistency_score,
            "checked_claims": len(consistency_checks),
            "claim_checks": consistency_checks
        }
        
        return consistency_report, consistency_score
    
    except Exception as e:
        raise Exception(f"Failed to check narrative consistency: {str(e)}")

def extract_factual_claims(narrative):
    """
    Extract factual claims from the narrative using rule-based NLP techniques
    
    Parameters:
    narrative (str): The financial narrative
    
    Returns:
    list: List of extracted factual claims
    """
    try:
        # Split the narrative into sentences
        sentences = sent_tokenize(narrative)
        
        # Define patterns for identifying factual claims
        price_pattern = re.compile(r'\$?\d+(?:\.\d+)?')
        percentage_pattern = re.compile(r'\d+(?:\.\d+)?\s*\%')
        date_pattern = re.compile(r'\d{4}-\d{2}-\d{2}')
        
        # Keywords for different claim types
        trend_keywords = ['increased', 'decreased', 'risen', 'fell', 'grew', 'declined', 'improved',
                         'deteriorated', 'higher', 'lower', 'upward', 'downward', 'bullish', 'bearish',
                         'outperformed', 'underperformed']
        
        volatility_keywords = ['volatile', 'stability', 'fluctuation', 'variability', 'deviation']
        
        comparison_keywords = ['compared to', 'relative to', 'versus', 'against', 'outperformed', 
                              'underperformed', 'better than', 'worse than', 'higher than', 'lower than']
        
        # Initialize the claims list
        claims = []
        
        # Process each sentence to identify potential claims
        for sentence in sentences:
            # Skip sentences in headers (markdown headers)
            if sentence.startswith('#') or sentence.startswith('##') or sentence.startswith('###'):
                continue
                
            # Skip disclaimer or notes
            if 'disclaimer' in sentence.lower() or 'note:' in sentence.lower():
                continue
            
            # Check for price mentions
            if price_pattern.search(sentence):
                claim_type = 'price_claim'
                if any(keyword in sentence.lower() for keyword in trend_keywords):
                    claim_type = 'price_trend_claim'
                claims.append({'claim_text': sentence, 'claim_type': claim_type})
                continue
            
            # Check for percentage mentions
            if percentage_pattern.search(sentence):
                claim_type = 'percentage_claim'
                if any(keyword in sentence.lower() for keyword in trend_keywords):
                    claim_type = 'percentage_trend_claim'
                claims.append({'claim_text': sentence, 'claim_type': claim_type})
                continue
            
            # Check for date-specific claims
            if date_pattern.search(sentence):
                claims.append({'claim_text': sentence, 'claim_type': 'date_specific_claim'})
                continue
            
            # Check for trend statements
            if any(keyword in sentence.lower() for keyword in trend_keywords):
                claims.append({'claim_text': sentence, 'claim_type': 'trend_claim'})
                continue
            
            # Check for volatility statements
            if any(keyword in sentence.lower() for keyword in volatility_keywords):
                claims.append({'claim_text': sentence, 'claim_type': 'volatility_claim'})
                continue
            
            # Check for comparative statements
            if any(keyword in sentence.lower() for keyword in comparison_keywords):
                claims.append({'claim_text': sentence, 'claim_type': 'comparison_claim'})
                continue
        
        # Limit number of claims to avoid performance issues
        return claims[:15]  # Return at most 15 claims
    
    except Exception as e:
        print(f"Error extracting factual claims: {str(e)}")
        return []

def verify_claim_against_data(claim, financial_data):
    """
    Verify a factual claim against the financial data using rule-based techniques
    
    Parameters:
    claim (dict): Dictionary containing claim_text and claim_type
    financial_data (pandas.DataFrame): The financial data
    
    Returns:
    dict: Verification result with consistency score
    """
    try:
        claim_text = claim["claim_text"]
        claim_type = claim["claim_type"]
        
        # Initialize sentiment analyzer for detecting sentiment
        sia = SentimentIntensityAnalyzer()
        
        # Create financial summary for comparison
        financial_summary = create_financial_summary(financial_data)
        
        # Extract key financial data metrics
        # Date range
        start_date = financial_data['Date'].min().strftime('%Y-%m-%d')
        end_date = financial_data['Date'].max().strftime('%Y-%m-%d')
        
        # Price data
        first_close = financial_data['Close'].iloc[0]
        last_close = financial_data['Close'].iloc[-1]
        price_change = last_close - first_close
        price_change_pct = (price_change / first_close) * 100
        price_min = financial_data['Low'].min()
        price_max = financial_data['High'].max()
        
        # Volume data
        avg_volume = financial_data['Volume'].mean()
        max_volume = financial_data['Volume'].max()
        
        # Moving averages if available
        ma_20 = None
        ma_50 = None
        if 'MA_20' in financial_data.columns and 'MA_50' in financial_data.columns:
            ma_20 = financial_data['MA_20'].iloc[-1] if not pd.isna(financial_data['MA_20'].iloc[-1]) else None
            ma_50 = financial_data['MA_50'].iloc[-1] if not pd.isna(financial_data['MA_50'].iloc[-1]) else None
            
        # Volatility if available
        volatility = None
        if 'Volatility_20d' in financial_data.columns:
            volatility = financial_data['Volatility_20d'].iloc[-1] * 100 if not pd.isna(financial_data['Volatility_20d'].iloc[-1]) else None
        
        # Initialize consistency score
        consistency_score = 0.5  # Default neutral score
        verification_result = "unverified"
        explanation = "No specific data points found to verify this claim."
        
        # Extract numbers and percentages from the claim
        extracted_numbers = re.findall(r'\$?(\d+(?:\.\d+)?)', claim_text)
        extracted_percentages = re.findall(r'(\d+(?:\.\d+)?)\s*\%', claim_text)
        
        # Check claim type and verify against data
        if claim_type == 'price_claim' or claim_type == 'price_trend_claim':
            # Verify price-related claims
            for number in extracted_numbers:
                num_value = float(number)
                # Check if the price is within the range of actual prices
                if price_min * 0.9 <= num_value <= price_max * 1.1:
                    consistency_score = 0.9
                    verification_result = "verified"
                    explanation = f"The price value {num_value} is within the observed price range (${price_min:.2f} to ${price_max:.2f})."
                    break
                elif price_min * 0.7 <= num_value <= price_max * 1.3:
                    consistency_score = 0.7
                    verification_result = "partially verified"
                    explanation = f"The price value {num_value} is outside but close to the observed price range (${price_min:.2f} to ${price_max:.2f})."
                    break
                else:
                    consistency_score = 0.2
                    verification_result = "contradicted"
                    explanation = f"The price value {num_value} is significantly outside the observed price range (${price_min:.2f} to ${price_max:.2f})."
        
        elif claim_type == 'percentage_claim' or claim_type == 'percentage_trend_claim':
            # Verify percentage-related claims
            for percentage in extracted_percentages:
                pct_value = float(percentage)
                # Check if the percentage is close to the actual change percentage
                if abs(pct_value - abs(price_change_pct)) <= 2:  # Within 2% is considered accurate
                    consistency_score = 0.95
                    verification_result = "verified"
                    explanation = f"The percentage {pct_value}% closely matches the observed change of {abs(price_change_pct):.2f}%."
                    break
                elif abs(pct_value - abs(price_change_pct)) <= 5:  # Within 5% is partially accurate
                    consistency_score = 0.7
                    verification_result = "partially verified"
                    explanation = f"The percentage {pct_value}% is reasonably close to the observed change of {abs(price_change_pct):.2f}%."
                    break
                else:
                    consistency_score = 0.3
                    verification_result = "contradicted"
                    explanation = f"The percentage {pct_value}% is significantly different from the observed change of {abs(price_change_pct):.2f}%."
        
        elif claim_type == 'date_specific_claim':
            # Verify date-specific claims
            dates_in_claim = re.findall(r'\d{4}-\d{2}-\d{2}', claim_text)
            for date in dates_in_claim:
                if date >= start_date and date <= end_date:
                    consistency_score = 0.9
                    verification_result = "verified"
                    explanation = f"The date {date} falls within the analyzed period ({start_date} to {end_date})."
                    break
                else:
                    consistency_score = 0.1
                    verification_result = "contradicted"
                    explanation = f"The date {date} falls outside the analyzed period ({start_date} to {end_date})."
        
        elif claim_type == 'trend_claim':
            # Verify trend claims based on actual price direction
            uptrend_words = ['increase', 'increased', 'rise', 'risen', 'grew', 'growth', 'upward', 'higher', 'up', 'bullish']
            downtrend_words = ['decrease', 'decreased', 'fall', 'fell', 'decline', 'declined', 'downward', 'lower', 'down', 'bearish']
            
            # Determine if the claim suggests an uptrend or downtrend
            is_uptrend_claim = any(word in claim_text.lower() for word in uptrend_words)
            is_downtrend_claim = any(word in claim_text.lower() for word in downtrend_words)
            
            # Check if the claim matches actual trend
            actual_uptrend = price_change > 0
            
            if (is_uptrend_claim and actual_uptrend) or (is_downtrend_claim and not actual_uptrend):
                consistency_score = 0.9
                verification_result = "verified"
                explanation = f"The claimed trend direction matches the observed price movement ({price_change_pct:.2f}%)."
            else:
                consistency_score = 0.1
                verification_result = "contradicted"
                explanation = f"The claimed trend direction contradicts the observed price movement ({price_change_pct:.2f}%)."
        
        elif claim_type == 'volatility_claim':
            # Verify volatility claims
            high_volatility_words = ['high', 'significant', 'increased', 'substantial']
            low_volatility_words = ['low', 'decreased', 'minimal', 'limited', 'reduced']
            
            is_high_volatility_claim = any(word in claim_text.lower() for word in high_volatility_words)
            is_low_volatility_claim = any(word in claim_text.lower() for word in low_volatility_words)
            
            if volatility is not None:
                high_volatility_threshold = 2.0  # 2% daily volatility is considered high
                
                if (is_high_volatility_claim and volatility > high_volatility_threshold) or \
                   (is_low_volatility_claim and volatility <= high_volatility_threshold):
                    consistency_score = 0.9
                    verification_result = "verified"
                    explanation = f"The volatility claim is consistent with the observed volatility of {volatility:.2f}%."
                else:
                    consistency_score = 0.2
                    verification_result = "contradicted"
                    explanation = f"The volatility claim contradicts the observed volatility of {volatility:.2f}%."
        
        elif claim_type == 'comparison_claim':
            # For comparison claims, just check for general sentiment consistency
            # This is a simplified approach
            sentiment_scores = sia.polarity_scores(claim_text)
            is_positive_claim = sentiment_scores['compound'] > 0.1
            is_negative_claim = sentiment_scores['compound'] < -0.1
            
            is_positive_actual = price_change_pct > 0
            
            if (is_positive_claim and is_positive_actual) or (is_negative_claim and not is_positive_actual):
                consistency_score = 0.7  # Less confidence since it's a general check
                verification_result = "partially verified"
                explanation = "The sentiment of the comparison appears to generally match the observed performance."
            else:
                consistency_score = 0.3
                verification_result = "partially contradicted"
                explanation = "The sentiment of the comparison appears to contradict the observed performance."
        
        # Return the verification result
        return {
            "claim_text": claim_text,
            "claim_type": claim_type,
            "consistency_score": consistency_score,
            "verification_result": verification_result,
            "explanation": explanation
        }
    
    except Exception as e:
        print(f"Error verifying claim: {str(e)}")
        return {
            "claim_text": claim.get("claim_text", ""),
            "claim_type": claim.get("claim_type", ""),
            "consistency_score": 0.0,
            "verification_result": "error",
            "explanation": f"Error during verification: {str(e)}"
        }

def create_financial_summary(financial_data):
    """
    Create a summary of the financial data for verification
    
    Parameters:
    financial_data (pandas.DataFrame): The financial data
    
    Returns:
    str: Summary of the financial data
    """
    summary = []
    
    # Date range
    summary.append(f"Date Range: {financial_data['Date'].min().strftime('%Y-%m-%d')} to {financial_data['Date'].max().strftime('%Y-%m-%d')}")
    
    # Price range
    summary.append(f"Price Range: Low ${financial_data['Low'].min():.2f} to High ${financial_data['High'].max():.2f}")
    
    # Latest prices
    latest_date = financial_data['Date'].max()
    latest_row = financial_data[financial_data['Date'] == latest_date]
    summary.append(f"Latest Close (as of {latest_date.strftime('%Y-%m-%d')}): ${latest_row['Close'].iloc[0]:.2f}")
    
    # Price changes
    if len(financial_data) > 1:
        first_close = financial_data['Close'].iloc[0]
        last_close = financial_data['Close'].iloc[-1]
        absolute_change = last_close - first_close
        percent_change = (absolute_change / first_close) * 100
        summary.append(f"Overall Price Change: {absolute_change:.2f} ({percent_change:.2f}%)")
    
    # Volume
    avg_volume = financial_data['Volume'].mean()
    max_volume = financial_data['Volume'].max()
    summary.append(f"Average Volume: {avg_volume:.0f}")
    summary.append(f"Maximum Volume: {max_volume:.0f}")
    
    # Moving averages
    if 'MA_20' in financial_data.columns and 'MA_50' in financial_data.columns:
        last_ma20 = financial_data['MA_20'].iloc[-1]
        last_ma50 = financial_data['MA_50'].iloc[-1]
        if not pd.isna(last_ma20) and not pd.isna(last_ma50):
            summary.append(f"Latest 20-day Moving Average: ${last_ma20:.2f}")
            summary.append(f"Latest 50-day Moving Average: ${last_ma50:.2f}")
    
    # Volatility
    if 'Volatility_20d' in financial_data.columns:
        last_volatility = financial_data['Volatility_20d'].iloc[-1]
        if not pd.isna(last_volatility):
            summary.append(f"Latest 20-day Volatility: {last_volatility*100:.2f}%")
    
    # Return as a string
    return "\n".join(summary)

def compute_consistency_score(consistency_report):
    """
    Compute an overall consistency score based on the consistency report
    
    Parameters:
    consistency_report (dict): The consistency report from check_narrative_consistency
    
    Returns:
    float: Overall consistency score between 0 and 1
    """
    if not consistency_report or 'overall_score' not in consistency_report:
        return 0.0
    
    return consistency_report['overall_score']
