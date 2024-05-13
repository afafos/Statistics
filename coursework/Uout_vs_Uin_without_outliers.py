import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

data_dir = "rawData"
output_voltage_data = {}

for filename in os.listdir(data_dir):
    if filename.endswith(".dat"):
        input_voltage = filename.split("V")[0]

        filepath = os.path.join(data_dir, filename)
        data = pd.read_csv(filepath, sep='\s+', header=None)

        if input_voltage not in output_voltage_data:
            output_voltage_data[input_voltage] = []

        # quartiles = np.percentile(data[5], [25, 75])
        # iqr = quartiles[1] - quartiles[0]
        # lower_bound = quartiles[0] - 1.5 * iqr
        # upper_bound = quartiles[1] + 1.5 * iqr

        # print(f"Filename: {filename}")
        # print(f"IQR: {iqr}")
        # print(f"Lower bound: {lower_bound}")
        # print(f"Upper bound: {upper_bound}")
        # print()
        # data = data[(data[5] > lower_bound) & (data[5] < upper_bound)]

        output_voltage_data[input_voltage].extend(data[5].tolist())

sorted_output_voltage_data = sorted(output_voltage_data.items(), key=lambda x: float(x[0]))

medians = [np.median(voltage_data) for _, voltage_data in sorted_output_voltage_data]

input_voltages = [float(input_voltage) for input_voltage, _ in sorted_output_voltage_data]
X = sm.add_constant(input_voltages)  # Adding a constant
y = medians
model_ols = sm.OLS(y, X).fit()
predictions_ols = model_ols.predict(X)

model_orm = sm.RLM(y, X).fit()
predictions_orm = model_orm.predict(X)

# Output coefficients
print("OLS Regression Coefficients:")
print(model_ols.params)
print("\nRLM Regression Coefficients:")
print(model_orm.params)

# Plot medians and regression lines
plt.figure(figsize=(20, 10))
plt.scatter(input_voltages, medians, color='red', marker='o', label='Data', s=40)
plt.plot(input_voltages, predictions_ols, color='blue', linewidth=2, label='LSM Regression')
# plt.plot(input_voltages, predictions_orm, color='green', linewidth=2, label='LAD Regression')
plt.xlabel('Input voltage (V)', fontsize=30)
plt.ylabel('Median of Output voltage', fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Median of Output Voltage for Different Input Voltages', fontsize=35)
plt.legend(fontsize=20)
plt.grid(True)
plt.show()
