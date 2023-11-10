import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
import seaborn as sns

# import datetime


st.title("Stock Market Analysis App")
st.sidebar.header("User Input")
stock_symbol = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")
start = st.sidebar.text_input("Enter Start Date (YYYY-MM-DD):", "2022-01-01")
end = st.sidebar.text_input("Enter End Date (YYYY-MM-DD):", "2023-01-01")
# start="2010-01-01"
# end="2023-01-01"
def get_stock_data(stock_symbol, start_date, end_date):
    df = yf.download(stock_symbol, start=start_date, end=end_date)
    return df
df = get_stock_data(stock_symbol, start, end) 
# # -------- imporvement part





# -------------------to this end
st.subheader("Stock Price Data")
st.write(df.describe())
# VISUALISATION 
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling (100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100 = df.Close.rolling (100).mean()
ma200 = df.Close.rolling (200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)

# splitting the data in training and testing
data_training=pd.DataFrame(df['Close'][0: int (len (df) *0.70)])
data_testing = pd.DataFrame (df[ 'Close'][int (len (df)*0.70): int(len(df))])
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler (feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)
# # SPLITTING DATA into x_train and y_train 
# DONT NEED TRAINING PART WE HAVE ALREADY TRAINED THE MODEL 
# x_train = []
# y_train = []
# for i in range(100, data_training_array.shape[0]):
#     x_train.append(data_training_array[i-100: i])
#     y_train.append(data_training_array[i, 0])
    
# x_train,y_train=np.array(x_train),np.array(y_train)
# AND USING PRETRAINED MODEL 
# LOAD MY MODEL 
model = load_model('my_model.keras')
# model=load_model('keras_model.h5')
# TESTING PART 
past_100_days = data_training. tail (100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]): 
    x_test.append(input_data[i-100: i]) 
    y_test.append(input_data[i, 0])


x_test, y_test = np.array(x_test), np.array(y_test)

# making prediction 
y_predicted = model.predict(x_test)

scaler=scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor
#Final Graph
st.subheader('Predictions vs Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
# plt.show()
st.pyplot(fig2)

