# Stock Price Prediction with LSTM

This project predicts stock prices using an LSTM (Long Short-Term Memory) neural network and presents the results via a web interface built with Streamlit. The app allows users to visualize historical stock data, compare actual vs predicted stock prices, and explore future price predictions.

## Features

- **Real-Time Stock Data**: Fetches stock data from Yahoo Finance.
- **Interactive Visualization**: 
  - Historical stock price with moving averages (MA50, MA200).
  - Actual vs predicted stock price comparison.
  - Predictions for the next 100 business days.
- **Performance Metrics**: Displays model accuracy and RMSE for evaluation.

## Technologies

- **Python**: Primary language used.
- **Streamlit**: For creating the interactive user interface.
- **Keras (LSTM)**: For building and running the stock prediction model.
- **Plotly**: Used for creating interactive and visually appealing charts.
- **yfinance**: Fetches real-time stock price data from Yahoo Finance.
- **Scikit-learn**: Used for data scaling and performance evaluation (RMSE, MAPE).
