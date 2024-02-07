from flask import Flask, request, jsonify
from flask_cors import CORS  # Import the CORS module
import yfinance as yf
import pandas as pd
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow import Sequential
from tensorflow import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from keras.callbacks import EarlyStopping
import yfinance as yf
import plotly.graph_objs as go
import yfinance as yf
import pandas as pd


# Path to your saved model
model_path = 'LSTM_Model2.h5'

# Load the model
model = tf.load_model(model_path)

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# Fetching gold price data for the last year
gold_data = yf.Ticker("GC=F")  # GC=F is the futures ticker for gold
gold_prices = gold_data.history(period="1y")  # Fetching data for the last year

# Creating the plot
fig = go.Figure()

# Adding the line chart
fig.add_trace(go.Scatter(x=gold_prices.index,
              y=gold_prices['Close'], mode='lines+markers', name='Gold Price'))

# Updating the layout for better readability
fig.update_layout(
    title='Gold Prices Over the Last Year',
    xaxis_title='Date',
    yaxis_title='Price in USD',
    hovermode='x'  # Shows the hover info at the x-axis level for better readability
)

# Show the figure
fig.show()


# Download the last 10 days of gold price data
gold_data = yf.download("GC=F", period="11d", interval="1d")

# Assuming you're interested in the 'Adj Close' column for the LSTM model
new_data_df = gold_data[['Adj Close']]
print(new_data_df)

# Normalize the new data
scaler = MinMaxScaler().fit(new_data_df)
# Use the same scaler object used during training
new_data_scaled = scaler.transform(new_data_df)

# Assuming the new data is already a sequence that matches the training sequence length
# Reshape the new data for LSTM input
# Reshape to [1, 10, 1] if there's only 1 feature (Adj Close)
new_data_reshaped = new_data_scaled.reshape(
    1, new_data_scaled.shape[0], new_data_scaled.shape[1])

# Assuming 'scaler' is your MinMaxScaler object used during training
# and 'new_data_df' is your DataFrame with the new gold prices

# Normalize the new data
new_data_scaled = scaler.transform(new_data_df)

# Reshape the new data for LSTM input
# Adjust dimensions as per your model's requirements
new_data_reshaped = new_data_scaled.reshape(
    1, new_data_scaled.shape[0], new_data_scaled.shape[1])

# Now 'new_data_reshaped' is ready to be fed into your LSTM model for prediction

prediction = model.predict(new_data_reshaped)
prediction_original_scale = scaler.inverse_transform(prediction)
# print(prediction_original_scale)

# Define the API endpoint to predict gold price


@app.route('/predict-gold-price', methods=['POST'])
def predict_gold_price():
    data = request.json
    # Replace 'feature' with the actual feature name in your model
    features = [data['feature']]

    try:
        # Make predictions using the loaded model
        #prediction = model.predict([features])[0]
        prediction = prediction_original_scale
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
