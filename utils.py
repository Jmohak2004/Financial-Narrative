import streamlit as st
import re
import pandas as pd
import numpy as np
from datetime import datetime

def display_error(title, error_message):
    """
    Display an error message in the Streamlit app
    
    Parameters:
    title (str): Error title
    error_message (str): Detailed error message
    """
    st.error(f"**{title}**\n\n{error_message}")

def format_currency(value):
    """
    Format a numeric value as currency
    
    Parameters:
    value (float): The numeric value
    
    Returns:
    str: Formatted currency string
    """
    return f"${value:,.2f}"

def highlight_inconsistencies(narrative, consistency_report):
    """
    Highlight inconsistencies in the narrative based on the consistency report
    
    Parameters:
    narrative (str): The generated narrative text
    consistency_report (dict): The consistency report with claim checks
    
    Returns:
    str: HTML-formatted narrative with highlighted inconsistencies
    """
    if not consistency_report or 'claim_checks' not in consistency_report:
        return narrative
    
    # Create a copy of the narrative
    highlighted_narrative = narrative
    
    # Sort claims by consistency score (lowest first to avoid highlighting issues)
    sorted_claims = sorted(
        consistency_report['claim_checks'], 
        key=lambda x: x.get('consistency_score', 1.0)
    )
    
    # Define highlight colors based on consistency score
    def get_highlight_color(score):
        if score < 0.5:
            return "#FFCCCC"  # Red highlight for low consistency
        elif score < 0.75:
            return "#FFFFCC"  # Yellow highlight for medium consistency
        else:
            return "#CCFFCC"  # Green highlight for high consistency
    
    # Highlight each claim with appropriate color
    for claim in sorted_claims:
        claim_text = claim.get('claim_text', '')
        score = claim.get('consistency_score', 1.0)
        verification = claim.get('verification_result', '')
        explanation = claim.get('explanation', '')
        
        if claim_text and score < 0.9:  # Only highlight claims with score < 0.9
            # Escape regex special characters in the claim text
            escaped_claim = re.escape(claim_text)
            highlight_color = get_highlight_color(score)
            
            # Create tooltip with verification details
            tooltip = f"Score: {score:.2f}, Result: {verification}, Explanation: {explanation}"
            
            # Replace the claim text with highlighted version (with tooltip)
            highlighted_claim = f'<span title="{tooltip}" style="background-color: {highlight_color};">{claim_text}</span>'
            highlighted_narrative = highlighted_narrative.replace(claim_text, highlighted_claim)
    
    # Wrap in HTML paragraph tags for proper rendering
    return f'<p>{highlighted_narrative}</p>'

def parse_date_range(date_str):
    """
    Parse date range string into start and end dates
    
    Parameters:
    date_str (str): Date range string (e.g., "2023-01-01 to 2023-12-31")
    
    Returns:
    tuple: (start_date, end_date) as datetime objects
    """
    try:
        dates = date_str.split("to")
        start_date = pd.to_datetime(dates[0].strip())
        end_date = pd.to_datetime(dates[1].strip())
        return start_date, end_date
    except:
        # Default to last year if parsing fails
        end_date = datetime.now()
        start_date = end_date - pd.DateOffset(years=1)
        return start_date, end_date

def truncate_text(text, max_length=1000):
    """
    Truncate text to a maximum length with ellipsis
    
    Parameters:
    text (str): Text to truncate
    max_length (int): Maximum length
    
    Returns:
    str: Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def find_key_dates(financial_data):
    """
    Find key dates in the financial data (e.g., highest/lowest price days)
    
    Parameters:
    financial_data (pandas.DataFrame): DataFrame with financial data
    
    Returns:
    dict: Dictionary of key dates and their significance
    """
    key_dates = {}
    
    if len(financial_data) > 0:
        # Find date with highest price
        highest_price_idx = financial_data['High'].idxmax()
        highest_price_date = financial_data.loc[highest_price_idx, 'Date']
        highest_price = financial_data.loc[highest_price_idx, 'High']
        key_dates['highest_price'] = {
            'date': highest_price_date,
            'value': highest_price,
            'description': f"Highest price of {format_currency(highest_price)}"
        }
        
        # Find date with lowest price
        lowest_price_idx = financial_data['Low'].idxmin()
        lowest_price_date = financial_data.loc[lowest_price_idx, 'Date']
        lowest_price = financial_data.loc[lowest_price_idx, 'Low']
        key_dates['lowest_price'] = {
            'date': lowest_price_date,
            'value': lowest_price,
            'description': f"Lowest price of {format_currency(lowest_price)}"
        }
        
        # Find date with highest volume
        highest_volume_idx = financial_data['Volume'].idxmax()
        highest_volume_date = financial_data.loc[highest_volume_idx, 'Date']
        highest_volume = financial_data.loc[highest_volume_idx, 'Volume']
        key_dates['highest_volume'] = {
            'date': highest_volume_date,
            'value': highest_volume,
            'description': f"Highest trading volume of {highest_volume:,}"
        }
        
        # Find dates with significant price changes (>5% in a day)
        if 'Daily_Return' in financial_data.columns:
            significant_changes = financial_data[abs(financial_data['Daily_Return']) > 0.05]
            for idx, row in significant_changes.iterrows():
                change_pct = row['Daily_Return'] * 100
                direction = "increase" if change_pct > 0 else "decrease"
                key_dates[f'significant_change_{idx}'] = {
                    'date': row['Date'],
                    'value': change_pct,
                    'description': f"Significant price {direction} of {abs(change_pct):.2f}%"
                }
    
    return key_dates
