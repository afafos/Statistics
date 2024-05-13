import os
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

combined_data = pd.DataFrame()

directory = "rawData"
for filename in os.listdir(directory):
    if filename.startswith("0.45V") and filename.endswith(".dat"):
        file_path = os.path.join(directory, filename)
        data = pd.read_csv(file_path, sep='\s+', header=None)
        combined_data = pd.concat([combined_data, data[[0, 5]]], ignore_index=True)

combined_data.columns = ['Time', 'Output Voltage']

combined_data.sort_values(by='Time', inplace=True)

plt.figure(figsize=(20, 6))
plt.scatter(combined_data['Time'], combined_data['Output Voltage'], label='Data', s=5)
plt.xlabel('Time')
plt.ylabel('Output Voltage')
plt.title('Output Voltage vs Time')
plt.legend()

X = sm.add_constant(combined_data['Time'])
model_ols = sm.OLS(combined_data['Output Voltage'], X).fit()
plt.plot(combined_data['Time'], model_ols.predict(X), color='red', label='LSM')

model_rlm = sm.RLM(combined_data['Output Voltage'], X).fit()
plt.plot(combined_data['Time'], model_rlm.predict(X), color='green', label='LAD')

plt.legend()
plt.xlim(combined_data['Time'].min(), combined_data['Time'].max()*1.05)
plt.show()

# Output coefficients
print("OLS Regression Coefficients:")
print(model_ols.params)
print("\nRLM Regression Coefficients:")
print(model_rlm.params)