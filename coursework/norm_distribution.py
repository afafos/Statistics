import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def filtering_data():
    data_dir = "rawData"
    avg_output_voltage = {}
    filtered_data = []
    data = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".dat"):
            input_voltage = filename.split("V")[0]

            filepath = os.path.join(data_dir, filename)
            data = pd.read_csv(filepath, sep='\s+', header=None)

            mean = data[5].mean()
            std_dev = data[5].std()
            lower_bound = mean - 3 * std_dev
            upper_bound = mean + 3 * std_dev

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
    return filtered_data[5], data[5], df


def create_regression_graph(df):
    plt.plot(df['Входное напряжение'], df['Среднее выходное напряжение'], marker='o', linestyle='None', markersize=4)
    plt.xlabel('Input voltage (V)')
    plt.ylabel('Average output voltage')
    plt.title('Dependence of output voltage on input voltage')
    plt.grid(True)
    plt.show()


def create_hist(data, raw_flag):
    def save_data():
        bin_centers = (bins[:-1] + bins[1:]) / 2
        np.savetxt(f'histogram_data_for_latex_{("clean","raw")[raw_flag]}.csv', np.column_stack((bin_centers, n)),
                   delimiter=',', header='Bin_Centers, Frequency', comments='')

    normalized_data = (data - np.mean(data)) / np.std(data)

    plt.hist(data,
               bins=50,
               density=True, )
    plt.title(f'Гистограмма {("","не")[raw_flag]}отфильтрованных данных')
    plt.xlabel('Значение')
    plt.ylabel('Количество')
    plt.grid(True)
    plt.show()

    n, bins, patches = plt.hist(normalized_data,
              bins=30,
              density=True)

    save_data()

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, 0, 1)
    plt.plot(x, p, 'k', linewidth=2)
    plt.title(f'Гистограмма {("", "не")[raw_flag]}отфильтрованных данных')
    plt.xlabel('Значение')
    plt.ylabel('Количество')
    plt.grid(True)
    plt.show()



data_f, row_data, df_ = filtering_data()

create_hist(data_f, False)
create_hist(row_data, True)

create_regression_graph(df_)
