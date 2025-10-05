# Importing the libraries
import os
import math
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
plt.style.use('fivethirtyeight')
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from keras.optimizers import SGD
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import pickle
import base64
from tensorflow.keras.models import load_model
from io import BytesIO
import tempfile


# Some functions to help out with
def plot_predictions(test,predicted):
    import matplotlib.pyplot as plt

def plot_predictions(test, predicted):
    plt.figure(figsize=(10, 5))
    plt.plot(test, color='red', label='Real PWR')
    plt.plot(predicted, color='green', label='Predicted PWR')
    plt.title('Model 1-Day Prediction')
    plt.xlabel('Time')
    plt.ylabel('Power')
    plt.legend()
    st.pyplot(plt.gcf())
    plt.clf()


def return_rmse(test, predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    return rmse

def return_mae(test, predicted):
    mae = mean_absolute_error(test, predicted)
    return mae

def return_r2(test, predicted):
    r2 = r2_score(test, predicted)
    return r2


def accuracy_within_tolerance(y_true, y_pred, tolerance=0.05):
        """
        Calculates the percentage of predictions within a given percentage tolerance.
        
        Parameters:
            y_true (array-like): Actual values
            y_pred (array-like): Predicted values
            tolerance (float): Acceptable percentage error (e.g., 0.05 for 5%)

        Returns:
            accuracy (float): Accuracy as a percentage
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        # Avoid division by zero
        nonzero_indices = y_true != 0
        y_true = y_true[nonzero_indices]
        y_pred = y_pred[nonzero_indices]

        relative_errors = np.abs((y_pred - y_true) / y_true)
        within_tolerance = relative_errors <= tolerance
        accuracy = np.mean(within_tolerance) * 100

        print(f"Accuracy within ±{tolerance * 100:.1f}%: {accuracy:.2f}%")
        return accuracy


st.title("Load Forecasting (Testing)")

uploaded_files = st.file_uploader("Upload file", type=["xls", "xlsx"], accept_multiple_files=True)
file = st.file_uploader('Model file .h5 model', type='.h5')
if uploaded_files and file:
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp.write(file.read())
        tmp.flush()  # Make sure all data is written
        model = load_model(tmp.name, compile=False)


    dfs = []
    
    for uploaded_file in uploaded_files:
        df = pd.read_excel(uploaded_file, engine="openpyxl")
        st.write(f"Filename: {uploaded_file.name}")
        df = df[['Date N Time', 'M1_PWR', 'M2_PWR', 'M3_PWR']]
        # st.dataframe(df)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df['Date N Time'] = pd.to_datetime(combined_df['Date N Time'])
    combined_df.set_index('Date N Time', inplace=True)
    df2=combined_df
    df2 = df2.sort_values(by='Date N Time')
    option = st.selectbox(
        "Which Motor Power to predict ?",
        ("M1_PWR", "M2_PWR", "M3_PWR"),
    )

    if st.button("Start Prediction"):
        df_refined2=df2[option].resample('6min').mean()
        df_refined2.fillna(0, inplace=True)
        st.line_chart(df_refined2)
        
        test_set = df_refined2.values
        test_set=test_set.reshape((test_set.shape[0],1))
        sc = MinMaxScaler(feature_range=(-1,1))
        test_set_scaled = sc.fit_transform(test_set)
        inputs=test_set_scaled
        X_test = []
        for i in range(1000,len(inputs)):
            X_test.append(inputs[i-1000:i,0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
        predicted_data = model.predict(X_test)
        predicted_data = sc.inverse_transform(predicted_data)
        test_set2=test_set[-240::]
        predicted_data2=predicted_data[-240::]
        plot_predictions(test_set2,predicted_data2)
        st.write(option, "Predictions :")

        st.write("Root Mean Squared Error: ", return_rmse(test_set2, predicted_data2))

        st.write("Mean Absolute Error: ", return_mae(test_set2, predicted_data2))

        st.write("R² Score: ", r2_score(test_set2, predicted_data2))

        errors = np.abs(test_set2 - predicted_data2)
        st.write("Max error:", errors.max())
        st.write("Mean error:", errors.mean())
        st.write("Accuracy within +/-50%: ", accuracy_within_tolerance(test_set2, predicted_data2, tolerance=0.5))

        plt.figure(figsize=(12, 6))
        plt.plot(errors, label='Absolute Error')
        plt.title('Absolute Error Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Absolute Error')
        plt.legend()

        st.pyplot(plt.gcf())
        plt.clf()
