import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv("GOLD_data.csv")


df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.asfreq('D')  # daily frequency
df['Close (USD/oz)'] = df['Close (USD/oz)'].interpolate()  # fill gaps

price_col = 'Close (USD/oz)'

model = ARIMA(df[price_col], order=(1, 1, 1))
model_fit = model.fit()

# Forecast next 45 periods 
forecast_steps = 45
forecast_res = model_fit.get_forecast(steps=forecast_steps)
conf_int = forecast_res.conf_int()
forecast = forecast_res.predicted_mean

# Create a forecast DataFrame
last_date = df.index[-1]
next_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq='D')

forecast_df = pd.DataFrame({
    'Date': next_dates,
    'Forecasted_Price_USD_per_oz': forecast.values,
    'Lower_CI': conf_int.iloc[:, 0].values,
    'Upper_CI': conf_int.iloc[:, 1].values
})

print("\nForecasted Gold Prices (next 45 days):")
print(forecast_df)

forecast_df.to_csv("GOLD_Forecast.csv", index=False)
print("\nForecast saved to GOLD_Forecast.csv")

# Plot actual vs forecast
plt.figure(figsize=(12, 6))
plt.plot(df.index, df[price_col], label='Historical Gold Price', color='blue')
plt.plot(next_dates, forecast, label='Forecast', color='red', linestyle='--')
plt.fill_between(next_dates,
                 conf_int.iloc[:, 0],
                 conf_int.iloc[:, 1],
                 color='red', alpha=0.2, label='95% Confidence Interval')

plt.title("Gold Price Forecast (USD/oz)")
plt.xlabel("Date")
plt.ylabel("Price (USD/oz)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
