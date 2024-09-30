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
from itertools import product


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
        self.data['Datetime'] = pd.to_datetime(
        self.data['Date'] + ' ' + self.data['Time'],
        dayfirst=True, errors='coerce')
    
    def initialize_holidays(self):
        """Initialize the UK holidays after the Datetime column is created."""
        self.uk_holidays = holidays.UnitedKingdom(years=self.data['Datetime'].dt.year.unique())

    def clean_gross_sales(self):
        """Convert 'Gross Sales' to numeric and handle missing or incorrect values."""
        self.data['Gross Sales'] = pd.to_numeric(
            self.data['Gross Sales'].replace('[£,]', '', regex=True), errors='coerce'
        )
        
        # Fill missing values with linear interpolation
        self.data['Gross Sales'].interpolate(method='linear', inplace=True)
        
        # Drop any remaining NaN values
        self.data.dropna(subset=['Gross Sales'], inplace=True)
        

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
        
        # Forward-fill missing 'Gross Sales' values
        weekly_sales['Gross Sales'].fillna(method='ffill', inplace=True)
        #weekly_sales['Is Afternoon Peak'] = self.add_afternoon_peak()  # Add afternoon peak flag
        #weekly_sales['Is Weekend Peak'] = self.add_weekend_peak()  # Add weekend peak flag
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
        df.dropna(inplace=True)  # Drop NaN values from the decomposition
        df.reset_index(drop=True, inplace=True)  # Reset the index after dropping rows
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
        
        df.reset_index(drop=True, inplace=True)  # Reset the index after dropping rows
        
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
        
# Create a class for early stopping
class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    """

    def __init__(self, patience=10, min_delta=0.0001):
        """
        :param patience: Number of epochs to wait before stopping.
        :param min_delta: Minimum change in loss to qualify as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False  # Continue training
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Stop training
            return False  # Continue training       


# Step 3: Training and Evaluation Functions
def train_model(model, train_loader, val_loader, loss_function, optimizer, epochs=100, early_stopping=None):
    """
    Trains the LSTM model.

    :param model: The LSTM model to train.
    :param train_loader: DataLoader for training data.
    :param val_loader: DataLoader for validation data.
    :param loss_function: The loss function to optimize.
    :param optimizer: The optimizer for training.
    :param epochs: Number of training epochs.
    :param early_stopping: EarlyStopping object to prevent overfitting.
    :return: Lists of training and validation losses per epoch.
    """
    
    train_losses = []
    val_losses = []
    
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

        avg_train_loss = train_loss / len(train_loader)  # Average training loss for the epoch
        train_losses.append(avg_train_loss)  # Store the loss for plotting


        val_loss = 0 # Reset validation loss for each epoch
        
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for seq, labels in val_loader:
                seq, labels = seq.float(), labels.float()
                y_pred = model(seq)
                val_loss += loss_function(y_pred, labels.view(-1,1)).item()

        # Average val loss for the epoch
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)  # Store the validation loss for plotting

        # Early stopping
        if early_stopping is not None:
            if early_stopping(avg_val_loss):
                print(f'Early stopping at epoch {epoch}')
                break
            
        

        if epoch % 10 == 0:  # Print loss every 10 epochs
    
            print(f'Epoch {epoch}, Train Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}')
    
    return train_losses, val_losses
         
def evaluate_model(model, data_loader,scaler):
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
    
    # Rescale to original values
    predicted_sales = scaler.inverse_transform(predicted_sales)
    actual_sales = scaler.inverse_transform(actual_sales)

    # Calculate metrics
    mae = mean_absolute_error(actual_sales, predicted_sales)
    mse = mean_squared_error(actual_sales, predicted_sales)
    rmse = np.sqrt(mse)
    

    
    
    
    print("Predicted Sales Shape:", predicted_sales.shape)
    print("Actual Sales Shape:", actual_sales.shape)
    
    return mae, mse, rmse, predicted_sales, actual_sales


def predict_future(model, last_sequence, prediction_months, scaler):
    """
    Function to predict future values based on the last known sequence (weekly data).
    :param model: The trained LSTM model
    :param last_sequence: The last sequence of input data to start predictions from
    :param prediction_months: Number of future periods (weeks) to predict
    :param scaler: Scaler used to inverse transform the scaled values
    :return: Array of predicted values (weekly predictions)
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

def walk_forward_validation(data, sequence_length, model_params, loss_function, scaler, n_splits=2, epochs=100):
    """
    Performs walk-forward validation.

    :param data: Preprocessed data as NumPy array.
    :param sequence_length: Number of time steps in each input sequence.
    :param model_params: Dictionary of model parameters.
    :param loss_function: Loss function for training.
    :param scaler: Scaler for inverse transforming.
    :param n_splits: Number of folds for validation.
    :param epochs: Number of epochs for training.
    :return: Lists of performance metrics.
    """
    n_samples = len(data)
    fold_size = int(n_samples / (n_splits + 1))

    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    maes, mses, rmses = [], [], []

    for i in range(n_splits):
        print(f"Processing Fold {i + 1}")
        train_end = fold_size * (i + 1)
        val_start = train_end
        val_end = min(val_start + fold_size, n_samples)

        train_data = data[:train_end]
        val_data = data[val_start:val_end]

        train_dataset = SalesDataset(train_data, sequence_length)
        val_dataset = SalesDataset(val_data, sequence_length)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        with mlflow.start_run(nested=True):
            # Log hyperparameters
            for param, value in model_params.items():
                mlflow.log_param(param, value)
            mlflow.log_param("fold", i + 1)

            # Initialize model
            model = LSTMModel(
                input_size=model_params['input_size'],
                hidden_layer_size=model_params['hidden_layer_size'],
                output_size=1,
                num_layers=model_params['num_layers'],
                dropout=model_params['dropout']
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=model_params['learning_rate'])
            early_stopping = EarlyStopping(patience=5, min_delta=0.0001)

            # Train model
            train_losses, val_losses = train_model(
                model, train_loader, val_loader,
                loss_function, optimizer, epochs=epochs,
                early_stopping=early_stopping
            )

            # Plot loss curves
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Training Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.legend()
            plt.title('Loss Curves')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            # Log the figure to MLflow
            mlflow.log_figure(plt.gcf(), "loss_curves.png")
            plt.close()
            

            # Evaluate model
            mae, mse, rmse, predicted_sales, actual_sales = evaluate_model(model, val_loader, scaler)

            # Log metrics
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rmse", rmse)
            

            # Log model
            mlflow.pytorch.log_model(model, artifact_path="model")

            # Store metrics
            maes.append(mae)
            mses.append(mse)
            rmses.append(rmse)
            

    # Log average metrics
    mlflow.log_metric("avg_mae", np.mean(maes))
    mlflow.log_metric("avg_mse", np.mean(mses))
    mlflow.log_metric("avg_rmse", np.mean(rmses))
   

    return maes, mses, rmses



def grid_search_hyperparameters(hyperparameter_grid, data, sequence_length, loss_function, scaler, n_splits=2, epochs=100):
    """
    Perform grid search over the hyperparameter space and train/evaluate the model.
    
    :param hyperparameter_grid: Dictionary with lists of hyperparameters to tune.
    :param data: Preprocessed data as NumPy array.
    :param sequence_length: Number of time steps in each input sequence.
    :param loss_function: Loss function for training.
    :param scaler: Scaler for inverse transforming.
    :param n_splits: Number of folds for validation.
    :param epochs: Number of epochs for training.
    :return: Best hyperparameters and metrics.
    """
    
    # Generate all combinations of hyperparameters
    keys, values = zip(*hyperparameter_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    # Calculate input_size from the preprocessed data
    input_size = data.shape[1] - 1  # Number of features (excluding target)
    
    # Initialize best hyperparameters and metrics
    best_mae, best_params, best_rmse = float('inf'), None, float('inf')
    
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Ensure MLflow is properly set up
    
    for params in param_combinations:
        print(f"Testing hyperparameters: {params}")
        
        # Add input_size to the current params since it's constant
        params['input_size'] = input_size

        # Ensure that the previous run is ended before starting a new one
        if mlflow.active_run():
            mlflow.end_run()  # Close any active run

        # Start a new MLflow run
        with mlflow.start_run(nested=True):
            # Perform walk-forward validation
            maes, mses, rmses = walk_forward_validation(
                data=data,
                sequence_length=sequence_length,
                model_params=params,
                loss_function=loss_function,
                scaler=scaler,
                n_splits=n_splits,
                epochs=epochs
            )
        
            # Calculate average MAE and RMSE
            avg_mae = np.mean(maes)
            avg_rmse = np.mean(rmses)
        
            # Log metrics to MLflow
            mlflow.log_params(params)
            mlflow.log_metric("avg_mae", avg_mae)
            mlflow.log_metric("avg_rmse", avg_rmse)
            
            # Log the best hyperparameters
            if avg_mae < best_mae:
                best_mae = avg_mae
                best_params = params
                best_rmse = avg_rmse
                print(f"New best parameters: {params} with MAE: {best_mae} and RMSE: {best_rmse}")
        
        # Ensure the current run is ended after logging
        mlflow.end_run()

    # Log best parameters and metrics in a final run
    with mlflow.start_run():
        mlflow.log_param("best_hidden_layer_size", best_params['hidden_layer_size'])
        mlflow.log_param("best_num_layers", best_params['num_layers'])
        mlflow.log_param("best_dropout", best_params['dropout'])
        mlflow.log_param("best_learning_rate", best_params['learning_rate'])
        mlflow.log_param("best_input_size", best_params['input_size'])
        mlflow.log_metric("best_mae", best_mae)
        mlflow.log_metric("best_rmse", best_rmse)

    return best_params, best_mae, best_rmse
    


# Step 4: Main Function
if __name__ == '__main__':
    file_path = '/Users/giorgosziakas/Desktop/D2023.csv'  # Path to the sales data file
    
    # Initialize the data preprocessor
    preprocessor = DataPreprocessor(file_path)
    data, dates = preprocessor.preprocess()

    # Define sequence length for the LSTM model
    sequence_length = 12  # Number of time steps in each input sequence

    # Define hyperparameter grid for tuning
    hyperparameter_grid = {
        'hidden_layer_size': [64, 128, 256],
        'num_layers': [1, 2, 3],
        'dropout': [0.1, 0.2, 0.3],
        'learning_rate': [0.001, 0.01, 0.1]
    }
    
    loss_function = nn.MSELoss()  # Mean Squared Error loss
    
    # Perform grid search
    best_params, best_mae, best_rmse = grid_search_hyperparameters(
        hyperparameter_grid=hyperparameter_grid,
        data=data,
        sequence_length=sequence_length,
        loss_function=loss_function,
        scaler=preprocessor.scaler,
        n_splits=2,
        epochs=100
    )
    
    # Print the best hyperparameters and metrics
    print(f"Best Hyperparameters: {best_params}")
    print(f"Best MAE: {best_mae}")
    print(f"Best RMSE: {best_rmse}")

