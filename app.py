import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet.plot import plot_plotly, plot_components_plotly

# List of ticker symbols
ticker_symbols = ['RELIANCE', 'TCS', 'HDFCBANK', 'ICICIBANK', 'BHARTIARTL', 'SBIN', 'INFY', 'LICI', 'ITC', 'HINDUNILVR', 'LT', 'BAJFINANCE', 'HCLTECH', 'MARUTI', 'SUNPHARMA', 'ADANIENT', 'KOTAKBANK', 'TITAN', 'ONGC', 'TATAMOTORS', 'NTPC', 'AXISBANK', 'DMART', 'ADANIGREEN', 'ADANIPORTS', 'ULTRACEMCO', 'ASIANPAINT', 'COALINDIA', 'BAJAJFINSV', 'BAJAJ-AUTO', 'POWERGRID', 'NESTLEIND', 'WIPRO', 'M&M', 'IOC', 'JIOFIN', 'HAL', 'DLF', 'ADANIPOWER', 'JSWSTEEL', 'TATASTEEL', 'SIEMENS', 'IRFC', 'VBL', 'ZOMATO', 'PIDILITIND', 'GRASIM', 'SBILIFE', 'BEL', 'LTIM', 'TRENT', 'PNB', 'INDIGO', 'BANKBARODA', 'HDFCLIFE', 'ABB', 'BPCL', 'PFC', 'GODREJCP', 'TATAPOWER', 'HINDALCO', 'HINDZINC', 'TECHM', 'AMBUJACEM', 'INDUSINDBK', 'CIPLA', 'GAIL']

# Function to fetch stock data from Yahoo Finance
def fetch_stock_data(ticker_symbol, start_date, end_date):
    ticker_symbol = ticker_symbol +".NS"
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
    df = stock_data[['Adj Close']].reset_index()
    df = df.rename(columns={'Date': 'ds', 'Adj Close': 'y'})
    # df.to_csv(f"{ticker_symbol}.csv")
    return df

# Function to train the Prophet model
def train_prophet_model(df):
    model = Prophet()
    model.fit(df)
    return model

# Function to make the forecast
def make_forecast(model, periods):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast

# Function to calculate performance metrics
def calculate_performance_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse}

# Function to determine sentiment
def determine_sentiment(actual, predicted):
    if actual > predicted:
        sentiment = 'Negative'
    elif actual < predicted:
        sentiment = 'Positive'
    else:
        sentiment = 'Neutral'
    return sentiment


# Streamlit app
def main():
    st.title('Stock Prediction on NSE Stocks')

    # Set up the layout
    st.sidebar.header('User Input Parameters')
    ticker_symbol = st.sidebar.selectbox('Enter Ticker Symbol', options=ticker_symbols, index=0)

    # Dropdown for training period selection
    training_period = st.sidebar.selectbox('Select Training Period', 
                                            options=['1 week', '1 month', '1 year', '10 years'])

    # Calculate start date and end date based on training period
    if training_period == '1 week':
        start_date = pd.to_datetime('today') - pd.DateOffset(weeks=1)
    elif training_period == '1 month':
        start_date = pd.to_datetime('today') - pd.DateOffset(months=1)
    elif training_period == '1 year':
        start_date = pd.to_datetime('today') - pd.DateOffset(years=1)
    elif training_period == '10 years':
        start_date = pd.to_datetime('today') - pd.DateOffset(years=10)

    end_date = pd.to_datetime('today')

    # Fetching the data for the selected training period
    df = fetch_stock_data(ticker_symbol, start_date, end_date)

    # Dropdown for forecast horizon selection
    forecast_horizon = st.sidebar.selectbox('Forecast Horizon', 
                                            options=['Next day', 'Next week', 'Next month'],
                                            format_func=lambda x: x.capitalize())
    
    # Convert the selected horizon to days
    horizon_mapping = {'Next day': 1, 'Next week': 7, 'Next month': 30}
    forecast_days = horizon_mapping[forecast_horizon]

    if st.sidebar.button('Forecast Stock Prices'):
        with st.spinner('Training model...'):
            model = train_prophet_model(df)
            forecast = make_forecast(model, forecast_days)

        
        forecast_reversed = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].iloc[-forecast_days:].iloc[::-1]        

        st.markdown("""
            *The prediction was made using the Prophet forecasting model. The model was trained on historical stock data and used to forecast future prices based on the observed trends and patterns.*
        """)
        st.subheader(f'Forecast Summary for {ticker_symbol}')
        latest_forecast = forecast_reversed.iloc[0]

        # Last Stock Price details with sentiment indicator
        actual_last_price = df["y"].iloc[-1]
        predicted_last_price = latest_forecast['yhat']
        sentiment = determine_sentiment(actual_last_price, predicted_last_price)
        st.warning(f'The last available adjusted closing price for {ticker_symbol} on {end_date.strftime("%d %B %Y")} is **{actual_last_price:.2f}**.')

        if sentiment == 'Positive':
            st.success(f'Overall predication indicates positive sentiment.')
        elif sentiment == 'Negative':
            st.error(f'Overall predication indicates negative sentiment.')
        else:
            st.info(f'Overall predication indicates neutral sentiment.')        

        # Prediction details
        st.markdown(f"""
            **Prediction for {forecast_horizon.lower()}:**
            
            - **Date:** {latest_forecast['ds'].strftime("%d %B %Y")}
            - **Predicted Price:** {latest_forecast['yhat']:.2f}
            - **Lower Bound:** {latest_forecast['yhat_lower']:.2f}
            - **Upper Bound:** {latest_forecast['yhat_upper']:.2f}
        """)

        st.markdown(f"""
            **Find below the prediction Data for the {forecast_horizon.lower()}:**
            
        """)
        st.write(forecast_reversed)

        
        # Calculate performance metrics
        # Function to determine if performance metrics are in a good range
        def evaluate_performance_metrics(metrics):
            evaluation = {}
            evaluation['MAE'] = 'Good' if metrics['MAE'] < 0.05 * (df['y'].max() - df['y'].min()) else 'Not Good'
            evaluation['MSE'] = 'Good' if metrics['MSE'] < 0.1 * (df['y'].max() - df['y'].min())**2 else 'Not Good'
            evaluation['RMSE'] = 'Good' if metrics['RMSE'] < 0.1 * (df['y'].max() - df['y'].min()) else 'Not Good'
            return evaluation

        # Calculate performance metrics
        actual = df['y']
        predicted = forecast['yhat'][:len(df)]
        metrics = calculate_performance_metrics(actual, predicted)

        # Evaluate performance metrics
        evaluation = evaluate_performance_metrics(metrics)

        metrics = calculate_performance_metrics(actual, predicted)
        MAE =metrics['MAE']
        MSE = metrics['MSE']
        RMSE = metrics['RMSE']        
             

        # Display evaluation
        st.subheader('Performance Evaluation')
        st.write('The metrics below provide a quantitative measure of the model’s accuracy:')
        maecolor = "green" if evaluation["MAE"] == "Good" else "red"
        msecolor = "green" if evaluation["MSE"] == "Good" else "red"
        rmsecolor = "green" if evaluation["RMSE"] == "Good" else "red"
        
        st.markdown(f'- **Mean Absolute Error (MAE):** {MAE:.2f} - :{maecolor}[{"Good" if evaluation["MAE"] == "Good" else "Not good"}] ')
        st.markdown("(The average absolute difference between predicted and actual values.)")

        st.markdown(f'- **Mean Squared Error (MSE):** {MSE:.2f} - :{msecolor}[{"Good" if evaluation["MSE"] == "Good" else "Not good"}]  ')
        st.markdown("(The average squared difference between predicted and actual values.)")

        st.markdown(f'- **Root Mean Squared Error (RMSE):** {RMSE:.2f} - :{rmsecolor}[{"Good" if evaluation["RMSE"] == "Good" else "Not good"}] ')
        st.markdown("(The square root of MSE, which is more interpretable in the same units as the target variable.)")
        
        
        


# Run the main function
if __name__ == "__main__":
    main()