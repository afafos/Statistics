import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

data_dir = "rawData"
avg_output_voltage = {}

for filename in os.listdir(data_dir):
    if filename.endswith(".dat"):
        input_voltage = filename.split("V")[0]

        filepath = os.path.join(data_dir, filename)
        data = pd.read_csv(filepath, sep='\s+', header=None)

        quartiles = np.percentile(data[5], [25, 75])
        iqr = quartiles[1] - quartiles[0]
        lower_bound = quartiles[0] - 1.5 * iqr
        upper_bound = quartiles[1] + 1.5 * iqr

        filtered_data = data[(data[5] >= lower_bound) & (data[5] <= upper_bound)]
        avg_output = filtered_data[5].mean()

        if input_voltage not in avg_output_voltage:
            avg_output_voltage[input_voltage] = []
        avg_output_voltage[input_voltage].append(avg_output)

for input_voltage, output_voltages in avg_output_voltage.items():
    avg_output_voltage[input_voltage] = np.mean(output_voltages)

df = pd.DataFrame(list(avg_output_voltage.items()), columns=['Входное напряжение', 'Среднее выходное напряжение'])
df['Входное напряжение'] = pd.to_numeric(df['Входное напряжение'])
df.sort_values(by='Входное напряжение', inplace=True)

X = df[['Входное напряжение']]
X = sm.add_constant(X)
y = df['Среднее выходное напряжение']
model_mnk = sm.OLS(y, X).fit()
mnk_prediction = model_mnk.predict(X)

model_mnm = sm.RLM(y, X).fit()
mnm_prediction = model_mnm.predict(X)

plt.plot(df['Входное напряжение'], df['Среднее выходное напряжение'], marker='o', linestyle='None', markersize=4, color='red')
plt.plot(df['Входное напряжение'], mnk_prediction, color='blue', label='МНК')
plt.plot(df['Входное напряжение'], mnm_prediction, color='green', label='МНМ')
plt.xlabel('Input voltage (V)')
plt.ylabel('Average output voltage')
plt.title('Dependence of output voltage on input voltage')
plt.legend()
plt.grid(True)
plt.show()
