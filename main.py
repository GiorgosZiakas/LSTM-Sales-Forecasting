import pandas as pd 
import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick 
import holidays
from statsmodels.tsa.seasonal import STL
import mlflow
import mlflow.pytorch


# Step 1: Data Preprocessing
class SalesDataset(Dataset):
    """
    A Pytorch Dataset class to handle the sequence generation for LSTM model

    """
    
    def __init__(self, data, sequence_length):
        # Store the data and the sequence length as instance variables
        self.data = data  # The process data
        self.sequence_length = sequence_length  # Number of time steps in each input sequence
        
    def __len__(self):
         # The length of the dataset is reduced by sequence_length to account for the sequence generation
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        # Extract the sequence of features X and the target Y 
        seq_x = self.data[idx:idx + self.sequence_length, :-1].astype(np.float32)  # All columns except the last (target)
        seq_y = self.data[idx + self.sequence_length, -1].astype(np.float32)  # The last column (target)
        return seq_x, seq_y
    
class DataPreprocessor:
    """
    A class to handle the data preprocessing pipeline for sales forecasting.
    """

    def __init__(self, file_path):
        self.data = pd.read_csv(file_path, sep=';')
        self.scaler = MinMaxScaler()  # For scaling 'Gross Sales'
        self.standard_scaler = StandardScaler()  # For scaling other features

    def combine_datetime(self):
        """Combine 'Date' and 'Time' columns into a single 'Datetime' column."""
        self.data['Date'] = self.data['Date'].astype(str)
        self.data['Time'] = self.data['Time'].astype(str)
        self.data['Datetime'] = pd.to_datetime(self.data['Date'] + ' ' + self.data['Time'], format='%d.%m.%y %H:%M:%S')

    def initialize_holidays(self):
        """Initialize the UK holidays after the Datetime column is created."""
        self.uk_holidays = holidays.UnitedKingdom(years=self.data['Datetime'].dt.year.unique())

    def clean_gross_sales(self):
        """Convert 'Gross Sales' to numeric and handle missing or incorrect values."""
        self.data['Gross Sales'] = pd.to_numeric(
            self.data['Gross Sales'].replace('[£,]', '', regex=True), errors='coerce'
        )
        self.data['Gross Sales'].fillna(0, inplace=True)  # Filling NaNs with 0 for now

    def add_holiday_flags(self):
        """Add flags for Christmas and Easter, and flag seasonality peaks."""
        # Christmas: Mark November and December
        self.data['Is Christmas'] = self.data['Datetime'].dt.month.apply(lambda x: 1 if x in [11, 12] else 0)
        
        # Easter: Flagging using the UK holidays package
        self.data['Is Easter'] = self.data['Datetime'].apply(lambda x: 1 if x in self.uk_holidays else 0)
        
        # General seasonality peaks in November, December, May, and June
        self.data['Is Seasonality Peak'] = self.data['Datetime'].dt.month.apply(
            lambda x: 1 if x in [5, 6, 11, 12] else 0 # May, June, November, December seasonality peaks
        )
        
    '''def add_afternoon_peak(self):
        """Flag weeks where afternoon peak hours (14:00 - 17:00) had high sales."""
        # First, create a column that flags afternoon peaks on a daily basis
        self.data['Is Afternoon Peak'] = self.data['Datetime'].dt.hour.apply(
            lambda x: 1 if 14 <= x <= 17 else 0
        )
        # Resample to weekly: If there was any afternoon peak during the week, flag it
        weekly_afternoon_peak = self.data.resample('W', on='Datetime')['Is Afternoon Peak'].sum()
        weekly_afternoon_peak = (weekly_afternoon_peak > 0).astype(int)  # Convert to 1/0 flag
        weekly_afternoon_peak = weekly_afternoon_peak.fillna(0)  # Ensure no NaNs, fill with 0
        
        return weekly_afternoon_peak'''


        
    '''def add_weekend_peak(self):
        """Flag weeks where Friday or Saturday had peak sales."""
        self.data['Is Friday'] = self.data['Datetime'].dt.weekday.apply(lambda x: 1 if x == 4 else 0)  # Friday
        self.data['Is Saturday'] = self.data['Datetime'].dt.weekday.apply(lambda x: 1 if x == 5 else 0)  # Saturday
        
        # Resample: Flag if Friday or Saturday had any sales activity
        weekly_friday_sales = self.data.resample('W', on='Datetime')['Is Friday'].sum()
        weekly_saturday_sales = self.data.resample('W', on='Datetime')['Is Saturday'].sum()

        weekend_peak_flag = (weekly_friday_sales > 0) | (weekly_saturday_sales > 0)  # Convert to True/False
        weekend_peak_flag = weekend_peak_flag.astype(int)  # Convert True/False to 1/0
        weekend_peak_flag = weekend_peak_flag.fillna(0)  # Ensure no NaNs, fill with 0
        
        return weekend_peak_flag'''


    def resample_and_aggregate(self):
        """Resample data to monthly frequency and aggregate features."""
        weekly_sales = self.data.resample('W', on='Datetime').agg({
            'Gross Sales': 'sum',  # Sum sales for the week
            'Is Christmas': 'max',  # Keep the flag if Christmas occurred in the week
            'Is Easter': 'max',     # Same for Easter
            'Is Seasonality Peak': 'max'  # Peak season flag
        }).reset_index()

        weekly_sales['Gross Sales'].fillna(method='ffill', inplace=True)
        #weekly_sales['Is Afternoon Peak'] = self.add_afternoon_peak()  # Add afternoon peak flag
        #weekly_sales['Is Weekend Peak'] = self.add_weekend_peak()  # Add weekend peak flag
        print("Weekly Sales Data: First 5 rows")
        print(weekly_sales.head()) # Debug print
        
        return weekly_sales
    
    def add_time_features(self, df):
        """Add time-based features: Year, Week of Year, and seasonal peaks."""
        df['Year'] = df['Datetime'].dt.year
        df['Week'] = df['Datetime'].dt.isocalendar().week  # Week of the year
        df['Quarter'] = df['Datetime'].dt.quarter  # Keep quarterly trends

        df['Is Summer Peak'] = df['Week'].apply(lambda x: 1 if x in range(23, 27) else 0)  # Example: June peaks
        df['Is Winter Peak'] = df['Week'].apply(lambda x: 1 if x in range(48, 52) else 0)  # Example: Christmas peaks

        return df


    def decompose_sales(self, df):
        """Decompose sales into trend, seasonality, and residual using STL decomposition."""
        stl = STL(df['Gross Sales'], seasonal=51, period= 52)  # Seasonal window based on your sales cycle
        result = stl.fit()
        df['Trend'] = result.trend
        df['Seasonality'] = result.seasonal
        df['Residual'] = result.resid
        return df

    def create_lag_features(self, df):
        """Add lagged features to capture sales trends from previous weeks."""
        df['Lag_1'] = df['Gross Sales'].shift(1)  # Last week's sales
        df['Lag_4'] = df['Gross Sales'].shift(4)  # Sales from four weeks ago
        df['Lag_12'] = df['Gross Sales'].shift(12)  # Sales from 12 weeks ago (quarterly lag)
        
        # Check how many rows have NaN values after lag creation
        print(f"Before dropping NaNs: {df.shape[0]} rows")
        
        # Drop rows with NaN due to lagging, but make sure enough data remains
        df.dropna(inplace=True)
        
        # Check how many rows are left after dropping NaNs
        print(f"After dropping NaNs: {df.shape[0]} rows")
        
        return df
         

    def scale_features(self, df):
        """Scale numerical features to standardize the input for modeling."""
        numerical_features = ['Year', 'Week', 'Quarter', 'Is Christmas', 'Is Easter', 
                              'Is Seasonality Peak', 
                              'Is Summer Peak', 'Is Winter Peak']
        df[numerical_features] = self.standard_scaler.fit_transform(df[numerical_features])
        
        # Scaling Gross Sales separately
        df['Gross Sales'] = self.scaler.fit_transform(df[['Gross Sales']])

        return df

    def preprocess(self):
        """Run the full preprocessing pipeline."""
        self.combine_datetime()
        self.clean_gross_sales()
        self.initialize_holidays()
        self.add_holiday_flags()

        # Resample the data to weekly frequency and add features
        weekly_sales = self.resample_and_aggregate()
        # Decompose sales into trend, seasonality, and residual components
        weekly_sales = self.decompose_sales(weekly_sales)

        weekly_sales = self.add_time_features(weekly_sales)
        weekly_sales = self.create_lag_features(weekly_sales)
        weekly_sales = self.scale_features(weekly_sales)
        
       # Be sure that the data is preprocessed correctly
        print("Data Preprocessing Completed. First 5 rows of preprocessed data:")
        print(weekly_sales.head()) 

        # Convert the relevant features to a numpy array for the model input
        self.data = weekly_sales[['Year', 'Week', 'Quarter', 'Is Summer Peak', 'Is Winter Peak', 
                                   'Is Christmas', 'Is Easter', 'Is Seasonality Peak', 
                                    'Trend', 'Seasonality', 'Residual', 
                                   'Lag_1', 'Lag_4', 'Lag_12', 'Gross Sales']].values

        return self.data, weekly_sales['Datetime']
    
# Step 2: Pytorch LSTM Model in OOP

class LSTMModel(nn.Module):
    """
    A LSTM model for time series forecasting
    
    """
    
    def __init__(self, input_size, hidden_layer_size,  output_size, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        
        # LSTM layer with specified number of input features, hidden units, and layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True, dropout=dropout)

        # A fully connected layer to produce the output from LSTM's hidden state
        self.linear = nn.Linear(hidden_layer_size, output_size)
        
    def forward(self, input_seq):
        # Pass the input through the LSTM layer
        lstm_out, _ = self.lstm(input_seq)
            
        # Pass the last time step's output from LSTM through the linear layer
        predictions = self.linear(lstm_out[:, -1])
        return predictions
        
        


# Step 3: Training and Evaluation Functions
def train_model(model, train_loader, val_loader, loss_function, optimizer, epochs=50):
    """
    Function to train the LSTM model.
    """
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        
        train_loss = 0  # Reset train loss for each epoch
        
        for seq, labels in train_loader:
            seq, labels = seq.float(), labels.float()  # Convert to float tensors
            optimizer.zero_grad()  # Clear previous gradients
            y_pred = model(seq)  # Make predictions
            loss = loss_function(y_pred, labels.view(-1,1))  # Calculate loss
            loss.backward()  # Backpropagate the error
            optimizer.step()  # Update model weights
            train_loss += loss.item()  # Add the loss to the training set loss

        val_loss = 0 # Reset validation loss for each epoch
        
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for seq, labels in val_loader:
                seq, labels = seq.float(), labels.float()
                y_pred = model(seq)
                val_loss += loss_function(y_pred, labels.view(-1,1)).item()

        # Average loss for the epoch
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        if epoch % 10 == 0:  # Print loss every 10 epochs
            print(f'Epoch {epoch}, Train Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}')

         
def evaluate_model(model, data_loader):
    """
    Function to evaluate the LSTM model.
    """
    model.eval()  # Set the model to evaluation mode
    predictions, actuals = [], []
    with torch.no_grad():  # Disable gradient calculation
        for seq, labels in data_loader:
            seq, labels = seq.float(), labels.float()  # Convert to float tensors
            
            # Debug print the shape of the sequence (batch size, sequence length, input size)
            print(f"Batch size during evaluation: {seq.shape}")
        
            y_pred = model(seq)  # Make predictions
            
            # Debug print the shape of the prediction and actual labels shapes (batch size, output size)
            print(f"Prediction Shape: {y_pred.shape}, Actual Shape: {labels.shape}")
            
            predictions.append(y_pred.numpy())  # Collect predictions
            actuals.append(labels.view(-1,1).numpy())  # Collect actual values
    predicted_sales = np.concatenate(predictions)
    actual_sales = np.concatenate(actuals)
    
    
    
    print("Predicted Sales Shape:", predicted_sales.shape)
    print("Actual Sales Shape:", actual_sales.shape)
    
    return predicted_sales, actual_sales 


def predict_future(model, last_sequence, prediction_months, scaler):
    """
    Function to predict future values based on the last known sequence (monthly data).
    :param model: The trained LSTM model
    :param last_sequence: The last sequence of input data to start predictions from
    :param prediction_months: Number of future periods (months) to predict
    :param scaler: Scaler used to inverse transform the scaled values
    :return: Array of predicted values (monthly predictions)
    """
    model.eval()  # Set model to evaluation mode
    predictions = []
    current_seq = last_sequence.clone()  # Clone the last sequence to avoid modifying the original data

    for _ in range(prediction_months):
        with torch.no_grad():
            # Predict the next value based on the current sequence (monthly data)
            prediction = model(current_seq).item()
            predictions.append(prediction)

            # Convert the prediction into the correct shape to concatenate with the current sequence
            prediction_tensor = torch.tensor([[[prediction]]]).float()  # Ensure it's in the right format

            # Create a new feature vector for the next time step
            new_features = torch.zeros((1, 1, current_seq.shape[2]))  # Placeholder for features
            new_features[0, 0, -1] = prediction  # Set the predicted 'Gross Sales'

            # Update the sequence by removing the oldest value and adding the new prediction
            new_seq = torch.cat((current_seq[:, 1:, :], new_features), dim=1)
            current_seq = new_seq

    # Inverse transform the predictions to original scale (denormalizing the predictions)
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    # add print fro debuggin
    print(f"Future Predictions: {predictions}")
    return predictions


def walk_forward_validation(data, sequence_length, model, loss_function, optimizer, scaler, n_splits=2, epochs=50, learning_rate=0.001, hidden_layer_size=128, num_layers=2, dropout=0.2):
    """
    Perform walk-forward validation on the time series data with MLflow logging.
    """
    n_samples = len(data)
    fold_size = int(n_samples / (n_splits + 1))

    # Start MLflow run outside the loop to track the entire walk-forward validation process
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    for i in range(n_splits):
        print(f"Processing Fold {i + 1}")
        train_end = fold_size * (i + 1)
        val_start = train_end
        val_end = min(val_start + fold_size, n_samples)

        # Split data
        train_data = data[:train_end]
        val_data = data[val_start:val_end]
        
        # Create datasets for PyTorch
        train_dataset = SalesDataset(train_data, sequence_length)
        val_dataset = SalesDataset(val_data, sequence_length)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Start a new MLflow run for each fold
        with mlflow.start_run(nested=True):
            # Log hyperparameters for this fold
            mlflow.log_param("sequence_length", sequence_length)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("hidden_layer_size", hidden_layer_size)
            mlflow.log_param("num_layers", num_layers)
            mlflow.log_param("dropout", dropout)
            mlflow.log_param("fold", i + 1)

            # Train the model for this fold
            train_model(model, train_loader, val_loader, loss_function, optimizer, epochs=epochs)

            # Evaluate the model on the validation set
            predicted_sales, actual_sales = evaluate_model(model, val_loader)

            # Rescale the predicted and actual sales to original values
            predicted_sales = scaler.inverse_transform(predicted_sales.reshape(-1, 1)).flatten()
            actual_sales = scaler.inverse_transform(actual_sales.reshape(-1, 1)).flatten()

            # Calculate accuracy metrics
            mae = mean_absolute_error(actual_sales, predicted_sales)
            mse = mean_squared_error(actual_sales, predicted_sales)
            rmse = np.sqrt(mse)

            # Log fold-specific metrics to MLflow
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rmse", rmse)

            # Reinitialize the model for each fold
            model = LSTMModel(input_size=model.lstm.input_size, hidden_layer_size=model.hidden_layer_size, output_size=1)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Log overall metrics for the run after all folds are done
    mlflow.log_metric("avg_mae", np.mean([mae]))
    mlflow.log_metric("avg_mse", np.mean([mse]))
    mlflow.log_metric("avg_rmse", np.mean([rmse]))

    return mae, mse, rmse

# Step 4: Main Function
if __name__ == 'main':
    file_path = '/Users/giorgosziakas/Desktop/D2023.csv'  # Path to the sales data file
    
    # Initialize the data preprocessor
    preprocessor = DataPreprocessor(file_path)
    data, dates = preprocessor.preprocess()

    # Define sequence length for the LSTM model
    sequence_length = 12  # Number of time steps in each input sequence

    # Initialize the LSTM model, loss function, and optimizer
    input_size = data.shape[1] - 1  # Number of input features (excluding the target)
    model = LSTMModel(input_size=input_size, hidden_layer_size=128, output_size=1, num_layers=2)
    loss_function = nn.MSELoss()  # Mean Squared Error loss for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

       
        
    # Perform walk-forward validation
    mae_scores, mse_scores, rmse_scores = walk_forward_validation(
        data=data,
        sequence_length=sequence_length,
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        scaler =preprocessor.scaler,
        n_splits=2
    )



   #Calculate and print average performance metrics
    avg_mae = np.mean(mae_scores)
    avg_mse = np.mean(mse_scores)
    avg_rmse = np.mean(rmse_scores)
    

    print(f"Average MAE: {avg_mae}")
    print(f"Average MSE: {avg_mse}")
    print(f"Average RMSE: {avg_rmse}")
  
    
    # Log average metrics to MLflow
    mlflow.log_metric('avg_mae', avg_mae)
    mlflow.log_metric('avg_mse', avg_mse)
    mlflow.log_metric('avg_rmse', avg_rmse)
    