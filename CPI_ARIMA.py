import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv("cpi_data.csv")

df['Date'] = df['Date'].astype(str).str.strip()

df = df.dropna(subset=['Date', 'South Africa Consumer Price Index (CPI) YoY'])

df['South Africa Consumer Price Index (CPI) YoY'] = df['South Africa Consumer Price Index (CPI) YoY'].str.rstrip('%').astype(float)

df['Date'] = df['Date'].apply(lambda x: pd.Period(x, freq='M'))

df = df.sort_values('Date')
df.set_index('Date', inplace=True)

model = ARIMA(df['South Africa Consumer Price Index (CPI) YoY'], order=(1, 1, 1))
model_fit = model.fit()

# Forecast next 4 months
last_period = df.index[-1]
next_periods = pd.period_range(start=last_period + 1, periods=4, freq='M')

forecast_res = model_fit.get_forecast(steps=4)
forecast = forecast_res.predicted_mean
conf_int = forecast_res.conf_int()

# Combine forecast results
forecast_df = pd.DataFrame({
    'Date': next_periods.astype(str),
    'Forecasted_CPI_YoY': forecast.values,
    'Lower_CI': conf_int.iloc[:, 0].values,
    'Upper_CI': conf_int.iloc[:, 1].values
})

print("\nForecasted CPI YoY (next 4 months):")
print(forecast_df)

forecast_df.to_csv("CPI_Forecast.csv", index=False)
print("\nForecast saved")

# Plot actual vs forecast
plt.figure(figsize=(10, 6))
plt.plot(df.index.to_timestamp(), df['South Africa Consumer Price Index (CPI) YoY'], 
         label='Historical CPI YoY', color='blue')
plt.plot(next_periods.to_timestamp(), forecast, 
         label='Forecast', color='red', linestyle='--')
plt.fill_between(next_periods.to_timestamp(),
                 conf_int.iloc[:, 0],
                 conf_int.iloc[:, 1],
                 color='red', alpha=0.2, label='95% Confidence Interval')

plt.title("Monthly South Africa CPI YoY Forecast")
plt.xlabel("Month")
plt.ylabel("CPI YoY (%)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
