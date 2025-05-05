import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_stock_chart(data):
    """
    Create an interactive chart based on the provided data
    
    Parameters:
    data (pandas.DataFrame): DataFrame containing stock data or general data
    
    Returns:
    plotly.graph_objects.Figure: Interactive chart
    """
    # Check if we have financial data (OHLCV) or general data
    is_financial_data = all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
    
    if is_financial_data:
        # Create subplot with 2 rows for financial data
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.1, 
                           subplot_titles=('Price', 'Volume'),
                           row_heights=[0.7, 0.3])
        
        # Add candlestick chart for price data
        fig.add_trace(
            go.Candlestick(
                x=data['Date'],
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="OHLC"
            ),
            row=1, col=1
        )
        
        # Add moving averages if available
        if 'MA_20' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data['Date'],
                    y=data['MA_20'],
                    line=dict(color='rgba(255, 165, 0, 0.7)', width=1.5),
                    name="20-day MA"
                ),
                row=1, col=1
            )
        
        if 'MA_50' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data['Date'],
                    y=data['MA_50'],
                    line=dict(color='rgba(46, 139, 87, 0.7)', width=1.5),
                    name="50-day MA"
                ),
                row=1, col=1
            )
        
        # Add volume chart
        fig.add_trace(
            go.Bar(
                x=data['Date'],
                y=data['Volume'],
                marker=dict(color='rgba(0, 0, 255, 0.5)'),
                name="Volume"
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f"{data['Symbol'].iloc[0] if 'Symbol' in data.columns else 'Stock'} Price Chart",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=600,
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
    else:
        # For generic data, create different visualizations based on the columns
        has_date = 'Date' in data.columns
        
        # Find numeric columns for plotting
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            # If no numeric columns are found, create a simple table view
            fig = go.Figure(data=[go.Table(
                header=dict(values=list(data.columns),
                            fill_color='paleturquoise',
                            align='left'),
                cells=dict(values=[data[col] for col in data.columns],
                          fill_color='lavender',
                          align='left'))
            ])
            fig.update_layout(
                title="Data Overview",
                height=600,
                margin=dict(l=40, r=40, t=60, b=40)
            )
        elif has_date and len(numeric_cols) > 0:
            # If we have date and numeric columns, create a time series plot
            fig = go.Figure()
            
            # Limit to at most 5 numeric columns for clarity
            for col in numeric_cols[:5]:
                fig.add_trace(
                    go.Scatter(
                        x=data['Date'],
                        y=data[col],
                        mode='lines',
                        name=col
                    )
                )
            
            fig.update_layout(
                title="Time Series Data",
                xaxis_title="Date",
                yaxis_title="Value",
                height=600,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=60, b=40)
            )
        else:
            # Create a combined visualization - scatter plot matrix or bar charts
            if len(numeric_cols) == 1:
                # Single numeric column - bar chart
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=list(range(len(data))),
                        y=data[numeric_cols[0]],
                        name=numeric_cols[0]
                    )
                )
                fig.update_layout(
                    title=f"{numeric_cols[0]} Values",
                    xaxis_title="Record Index",
                    yaxis_title=numeric_cols[0],
                    height=600,
                    margin=dict(l=40, r=40, t=60, b=40)
                )
            elif len(numeric_cols) == 2:
                # Two numeric columns - scatter plot
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=data[numeric_cols[0]],
                        y=data[numeric_cols[1]],
                        mode='markers',
                        name=f"{numeric_cols[0]} vs {numeric_cols[1]}"
                    )
                )
                fig.update_layout(
                    title=f"{numeric_cols[0]} vs {numeric_cols[1]}",
                    xaxis_title=numeric_cols[0],
                    yaxis_title=numeric_cols[1],
                    height=600,
                    margin=dict(l=40, r=40, t=60, b=40)
                )
            else:
                # Multiple numeric columns (>2) - parallel coordinates
                dimensions = [dict(range=[data[col].min(), data[col].max()],
                                label=col, values=data[col]) for col in numeric_cols[:6]]
                
                fig = go.Figure(go.Parcoords(
                    line=dict(color='blue'),
                    dimensions=dimensions
                ))
                
                fig.update_layout(
                    title="Parallel Coordinates Plot",
                    height=600,
                    margin=dict(l=80, r=80, t=60, b=40)
                )
    
    return fig

def create_market_trend_chart(data):
    """
    Create an interactive chart showing market trends or general data trends
    
    Parameters:
    data (pandas.DataFrame): DataFrame containing market index data or any dataset
    
    Returns:
    plotly.graph_objects.Figure: Interactive trend chart
    """
    fig = go.Figure()
    
    # Check if we have market data or general data
    has_date = 'Date' in data.columns
    has_close = 'Close' in data.columns
    is_market_data = has_date and has_close
    
    if is_market_data:
        # Create market trend chart
        fig.add_trace(
            go.Scatter(
                x=data['Date'],
                y=data['Close'],
                mode='lines',
                name=data['Index'].iloc[0] if 'Index' in data.columns else 'Market Index',
                line=dict(color='rgb(0, 76, 153)', width=2)
            )
        )
        
        # Add moving averages if available
        if 'MA_20' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data['Date'],
                    y=data['MA_20'],
                    mode='lines',
                    line=dict(color='rgba(255, 165, 0, 0.7)', width=1.5, dash='dot'),
                    name="20-day MA"
                )
            )
        
        if 'MA_50' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data['Date'],
                    y=data['MA_50'],
                    mode='lines',
                    line=dict(color='rgba(46, 139, 87, 0.7)', width=1.5, dash='dash'),
                    name="50-day MA"
                )
            )
        
        # Update layout
        fig.update_layout(
            title=f"{data['Index'].iloc[0] if 'Index' in data.columns else 'Market Index'} Trend",
            xaxis_title="Date",
            yaxis_title="Index Value",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40)
        )
    else:
        # For general data, create a trend or distribution visualization
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        
        if has_date and numeric_cols:
            # Create a line chart with the first numeric column vs date
            fig.add_trace(
                go.Scatter(
                    x=data['Date'],
                    y=data[numeric_cols[0]],
                    mode='lines+markers',
                    name=numeric_cols[0],
                    line=dict(color='rgb(0, 76, 153)', width=2)
                )
            )
            
            # Add a second numeric column if available
            if len(numeric_cols) > 1:
                fig.add_trace(
                    go.Scatter(
                        x=data['Date'],
                        y=data[numeric_cols[1]],
                        mode='lines+markers',
                        name=numeric_cols[1],
                        line=dict(color='rgb(204, 0, 0)', width=2)
                    )
                )
            
            # Update layout
            fig.update_layout(
                title="Data Trends Over Time",
                xaxis_title="Date",
                yaxis_title="Value",
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=60, b=40)
            )
        elif len(numeric_cols) > 0:
            # Create a box plot for numerical distributions
            box_data = []
            for col in numeric_cols[:5]:  # Limit to 5 columns
                box_data.append({
                    'y': data[col],
                    'type': 'box',
                    'name': col
                })
            
            fig = go.Figure(data=box_data)
            
            # Update layout
            fig.update_layout(
                title="Data Distribution",
                yaxis_title="Value",
                height=400,
                boxmode='group',
                margin=dict(l=40, r=40, t=60, b=40)
            )
        else:
            # Create a simple bar chart of value counts for the first categorical column
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            if categorical_cols:
                col = categorical_cols[0]
                value_counts = data[col].value_counts().reset_index()
                value_counts.columns = [col, 'Count']
                
                fig.add_trace(
                    go.Bar(
                        x=value_counts[col],
                        y=value_counts['Count'],
                        name='Count'
                    )
                )
                
                # Update layout
                fig.update_layout(
                    title=f"{col} Distribution",
                    xaxis_title=col,
                    yaxis_title="Count",
                    height=400,
                    margin=dict(l=40, r=40, t=60, b=40)
                )
            else:
                # Fallback if no suitable columns are found
                fig.add_annotation(
                    x=0.5,
                    y=0.5,
                    text="No suitable data for visualization",
                    showarrow=False,
                    font=dict(size=20)
                )
                
                # Update layout
                fig.update_layout(
                    height=400,
                    margin=dict(l=40, r=40, t=60, b=40)
                )
    
    return fig

def create_consistency_gauge(consistency_score, threshold=0.85):
    """
    Create a gauge chart displaying the consistency score
    
    Parameters:
    consistency_score (float): The consistency score (0 to 1)
    threshold (float): The threshold for acceptable consistency
    
    Returns:
    plotly.graph_objects.Figure: Gauge chart for consistency score
    """
    # Define color scale based on consistency level
    if consistency_score >= 0.9:
        color = "green"
        level = "Excellent"
    elif consistency_score >= 0.75:
        color = "lightgreen"
        level = "Good"
    elif consistency_score >= threshold:
        color = "yellow"
        level = "Acceptable"
    elif consistency_score >= 0.5:
        color = "orange"
        level = "Questionable"
    else:
        color = "red"
        level = "Poor"
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=consistency_score * 100,  # Convert to percentage
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Consistency Score", "font": {"size": 24}},
        delta={"reference": threshold * 100, "increasing": {"color": "green"}, "decreasing": {"color": "red"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkblue"},
            "bar": {"color": color},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "gray",
            "steps": [
                {"range": [0, 50], "color": "red"},
                {"range": [50, threshold * 100], "color": "orange"},
                {"range": [threshold * 100, 90], "color": "lightgreen"},
                {"range": [90, 100], "color": "green"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": threshold * 100
            }
        }
    ))
    
    # Add annotation for level
    fig.add_annotation(
        x=0.5,
        y=0.25,
        text=f"Level: {level}",
        showarrow=False,
        font=dict(size=16)
    )
    
    # Update layout
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig
