import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


df = pd.read_csv("gdp_data.csv")

df['Date'] = df['Date'].astype(str).str.strip()  # ensure it's a clean string
df = df.dropna(subset=['Date', 'GDP_growth_qoq_annualised'])  # drop missing rows

df['Date'] = df['Date'].apply(lambda x: pd.Period(x, freq='Q'))

df = df.sort_values('Date')
df.set_index('Date', inplace=True)

model = ARIMA(df['GDP_growth_qoq_annualised'], order=(1, 1, 1))
model_fit = model.fit()

# Forecast next 4 quarters 
last_period = df.index[-1]
next_periods = pd.period_range(start=last_period + 1, periods=4, freq='Q')

forecast_res = model_fit.get_forecast(steps=4)
forecast = forecast_res.predicted_mean
conf_int = forecast_res.conf_int()

# Combine forecast results
forecast_df = pd.DataFrame({
    'Date': next_periods.astype(str),
    'Forecasted_GDP_growth': forecast.values,
    'Lower_CI': conf_int.iloc[:, 0].values,
    'Upper_CI': conf_int.iloc[:, 1].values
})

print("\nForecasted GDP Growth (next 4 quarters):")
print(forecast_df)

forecast_df.to_csv("GDP_Forecast.csv", index=False)
print("\nForecast saved")

# Plot actual vs forecast
plt.figure(figsize=(10, 6))
plt.plot(df.index.to_timestamp(), df['GDP_growth_qoq_annualised'], label='Historical GDP Growth', color='blue')
plt.plot(next_periods.to_timestamp(), forecast, label='Forecast', color='red', linestyle='--')
plt.fill_between(next_periods.to_timestamp(),
                 conf_int.iloc[:, 0],
                 conf_int.iloc[:, 1],
                 color='red', alpha=0.2, label='95% Confidence Interval')

plt.title("Quarterly GDP Growth Forecast")
plt.xlabel("Quarter")
plt.ylabel("GDP Growth (QoQ Annualised, %)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
