import os
import pandas as pd
import matplotlib.pyplot as plt

data_dir = "rawData"
output_voltage_data = {}

for filename in os.listdir(data_dir):
    if filename.endswith(".dat"):
        input_voltage = filename.split("V")[0]

        filepath = os.path.join(data_dir, filename)
        data = pd.read_csv(filepath, sep='\s+', header=None)

        if input_voltage not in output_voltage_data:
            output_voltage_data[input_voltage] = []

        output_voltage_data[input_voltage].extend(data[5].tolist())


sorted_output_voltage_data = sorted(output_voltage_data.items(), key=lambda x: float(x[0]))


plt.figure(figsize=(20, 20))
boxplot_data = [voltage_data for _, voltage_data in sorted_output_voltage_data]
labels = [input_voltage for input_voltage, _ in sorted_output_voltage_data]
plt.boxplot(boxplot_data, labels=labels, boxprops=dict(linewidth=2), medianprops=dict(color='red'))
plt.xlabel('Input voltage (V)', fontsize=30)
plt.ylabel('Output voltage', fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Boxplot of output voltage for different input voltages', fontsize=35)
plt.grid(True)
plt.show()
