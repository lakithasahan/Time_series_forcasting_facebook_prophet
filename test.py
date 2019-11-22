import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
from fbprophet.plot import add_changepoints_to_plot

# Python

""""
df = pd.read_csv('prophet-master/examples/AAPL.csv')
data_ = df.copy()
drop_data=['Open','High','Low','Adj Close','Volume']
data_=data_.drop(drop_data,axis=1)
filtered_data=data_.rename(columns={"Date": "ds", "Close": "y"}, errors="raise")
"""

filtered_data = pd.read_csv('prophet-master/examples/example_wp_log_peyton_manning.csv')

print(len(filtered_data))

train_length = int(len(filtered_data) * 1)
print(train_length)
train_dataset = filtered_data[0:train_length]
print(train_dataset)

plt.plot(train_dataset['y'])
plt.show()

# Python
m = Prophet(yearly_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=0.2, interval_width=0.95)
m.fit(train_dataset)

future = m.make_future_dataframe(periods=365, freq='D')
print(future.columns)

forecast = m.predict(future)
print(forecast.columns)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
fig1 = m.plot(forecast)
plt.show()
fig2 = m.plot_components(forecast)
plt.show()
