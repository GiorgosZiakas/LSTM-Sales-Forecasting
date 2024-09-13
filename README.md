# This project implements a time series forecasting model using Long Short-Term Memory (LSTM) networks, 
which are a type of Recurrent Neural Network (RNN) particularly suited for sequence data. The model predicts weekly sales based on various features including holidays,
seasonality peaks, trends, and historical sales data. The main goal is to provide an accurate forecast to assist decision-makers in managing inventory, optimizing staffing, and improving sales strategies.

# Features
Data Preprocessing:

  Datetime Handling: Combines date and time columns into a unified datetime format.
  Holiday Flags: Adds binary flags for holidays such as Christmas and Easter.
  Seasonality Detection: Flags key seasonal peaks (e.g., May, June, November, and December).
  Sales Decomposition: Uses Seasonal-Trend decomposition to break down sales data into trend, seasonality, and residual components.
  Lag Features: Incorporates past sales data as input features (lags of 1 week, 4 weeks, and 12 weeks).
  
LSTM Model:

  The forecasting model is built using a Long Short-Term Memory (LSTM) network, a special type of Recurrent Neural Network (RNN) designed to capture long-term dependencies in sequence data.
  LSTM networks are particularly effective for time series forecasting tasks because they can remember patterns over long sequences of data, mitigating the vanishing gradient problem found in traditional RNNs.
  
  
Model Training and Validaion:

  The model is trained using walk-forward validation to evaluate performance over multiple time folds. This ensures robust testing on unseen data and prevents overfitting to specific periods.
  The model predicts sales for a specified future period and can be fine-tuned using various hyperparameters (e.g., hidden layer size, learning rate, number of LSTM layers).
  
Plotting:

Visualizes the actual vs predicted sales to help assess the model's performance and better understand forecasting accuracy over time.

# Clone the repository to your local machine
git clone https://github.com/yGiorgosZiakas/sales-forecasting-lstm.git

# Install the required Python packages
pip install -r requiremnents.txt


# Acknowledgements
This project was developed using PyTorch for deep learning, pandas for data manipulation, and matplotlib for plotting. 
Special thanks to the creators of these tools and libraries for enabling the development of such projects.
