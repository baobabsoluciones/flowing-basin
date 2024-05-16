"""
plot_prices.py
Plot the average price curve throughout a day
"""

import matplotlib.pyplot as plt
import pandas as pd

filename = f'price_charts/avg_price_curve.eps'

path_historical_data = "../data/history/historical_data_clean.pickle"
historical_data = pd.read_pickle(path_historical_data)
print(historical_data)

historical_data['datetime'] = pd.to_datetime(historical_data['datetime'])
historical_data['hour'] = historical_data['datetime'].dt.hour
hourly_avg_price = historical_data.groupby('hour')['price'].mean()

# Plot the average price curve
plt.figure(figsize=(8, 4))
hourly_avg_price.plot(kind='line', marker='o', color='red')
plt.title('Average Price Curve Throughout a Day (from November 2019 to November 2021)')
plt.xlabel('Hour of the Day')
plt.ylabel('Average Price (€/MWh)')
plt.grid(True)
plt.xticks(range(24))  # Show all hours
plt.tight_layout()
plt.savefig(filename, format='eps')
plt.show()

