# ARIMA Time Series
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('your_file.csv')
df['date'] = pd.to_datetime(df['date_column'])
df.set_index('date', inplace=True)
ts = df['value_column']  # Your time series column

# Plot original data
plt.figure(figsize=(12, 4))
plt.plot(ts)
plt.title('Original Time Series')
plt.show()

# Create and fit ARIMA model
# (p, d, q) = (autoregressive order, differencing, moving average order)
model = ARIMA(ts, order=(1, 1, 1))  # Try different values
fitted_model = model.fit()

print(fitted_model.summary())

# Forecast
forecast_steps = 10
forecast = fitted_model.forecast(steps=forecast_steps)
print("\nForecast:\n", forecast)

# Visualize forecast
plt.figure(figsize=(12, 4))
plt.plot(ts, label='Original')
plt.plot(forecast, label='Forecast', color='red')
plt.legend()
plt.title('ARIMA Forecast')
plt.show()


# SARIMA Time Series
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load data
df = pd.read_csv('your_file.csv')
df['date'] = pd.to_datetime(df['date_column'])
df.set_index('date', inplace=True)
ts = df['value_column']

# Create and fit SARIMA model
# (p,d,q) x (P,D,Q,s) where s is seasonal period
# For monthly data with yearly seasonality: s=12
# For daily data with weekly seasonality: s=7
model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
fitted_model = model.fit()

print(fitted_model.summary())

# Forecast
forecast_steps = 12
forecast = fitted_model.forecast(steps=forecast_steps)
print("\nForecast:\n", forecast)

# Visualize
plt.figure(figsize=(12, 4))
plt.plot(ts, label='Original')
plt.plot(forecast, label='Forecast', color='red')
plt.legend()
plt.title('SARIMA Forecast')
plt.show()

# Train-test split for evaluation
train_size = int(len(ts) * 0.8)
train, test = ts[:train_size], ts[train_size:]
model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12))
fitted = model.fit()
predictions = fitted.forecast(steps=len(test))
print("\nRMSE:", mean_squared_error(test, predictions, squared=False))

