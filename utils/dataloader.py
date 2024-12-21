import pandas as pd
import numpy as np
import re

from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class DataLoader(object):
    def fit(self, dataset):
        self.dataset = dataset.copy()

    # apply regex
    def calculate_mean_price(self, price):
        price = str(price)
        if ' - ' in price:
            lower, upper = price.split('-')
            return round((float(lower) + float(upper)) / 2)
        return float(price)

    def frequency_units(self, frequency):
        frequency = str(frequency)
        value, unit = frequency.split(' ')
        unit_map = {'GHz': 1, 'MHz': 0.001}
        multiplier = unit_map[unit]
        value = float(value) * multiplier
        return value

    def cache_type(self, cache):
        if pd.isnull(cache):
            return pd.NA, pd.NA
        cache = str(cache)
        if cache.count(' ') == 1:
            cache += ' L3'
        data = re.split(r'(\d+\.?\d*)\s(\w+)\s(.*)', cache)[1:-1]
        value, unit, c_type = data
        unit_map = {'MB': 1, 'KB': 0.001}
        multiplier = unit_map[unit]
        value = float(value) * multiplier
        return value, c_type

    def bus_type(self, bus_frequency):
        bus_frequency = str(bus_frequency)
        types = ['QPI', 'DMI', 'FSB', 'DMI3', 'PCIE']
        data = bus_frequency.split(' ')
        if data[-1].strip() in types:
            return data[-1]
        return pd.NA

    def memory_size(self, memory):
        if pd.isnull(memory):
            return pd.NA
        memory = str(memory)
        value, unit = memory.split(' ')
        unit_map = {'GB': 1, 'TB': 1024}
        multiplier = unit_map[unit]
        value = float(value) * multiplier
        return value

    def extract_temperature(self, temperature):
        temperature = str(temperature)
        if any(char in temperature for char in [',', ';', '=']):
            numbers = [float(num) for num in re.findall(r'\d+\.\d+|\d+', temperature)]
            return max(numbers) if numbers else pd.NA
        elif temperature != 'nan':
            match = re.search(r'\d+\.\d+|\d+', temperature)
            if isinstance(match, re.Match):
                return float(match.group()) if match else pd.NA
            elif isinstance(match, list):
                numbers = [float(num) for num in match]
                return max(numbers) if match else pd.NA
            else:
                return pd.NA

    def load_data(self):
        # Columns with no data, not useful for prediction or have too many missing values
        drop_elements = ['Processor_Number', 'Launch_Date', 'Max_Turbo_Frequency', 'Processor_Graphics_',
                         'Graphics_Base_Frequency', 'Graphics_Max_Dynamic_Frequency', 'Graphics_Video_Max_Memory',
                         'Graphics_Output', 'Support_4k', 'Max_Resolution_HDMI', 'Max_Resolution_DP',
                         'Max_Resolution_eDP_Integrated_Flat_Panel', 'DirectX_Support', 'OpenGL_Support',
                         'PCI_Express_Configurations_', 'Secure_Key']
        self.dataset = self.dataset.drop(drop_elements, axis=1)

        self.dataset['Processor_Base_Frequency'].fillna(self.dataset['Processor_Base_Frequency'].mode()[0], inplace=True)

        # Change Status just to Old and New (in binary)
        self.dataset['Status'] = self.dataset['Status'].map(
            {'End of Interactive Support': 0, 'End of Life': 0, 'Launched': 1, 'Announced': 1})

        self.dataset['Recommended_Customer_Price'] = self.dataset['Recommended_Customer_Price'].str.replace('$', '')
        self.dataset['Recommended_Customer_Price'] = self.dataset['Recommended_Customer_Price'].str.replace(',', '')
        # Change the price to the mean of the range
        self.dataset['Recommended_Customer_Price'] = self.dataset['Recommended_Customer_Price'].apply(
            self.calculate_mean_price)
        self.dataset['Recommended_Customer_Price'] = self.dataset['Recommended_Customer_Price'].astype(float)
        self.dataset['Recommended_Customer_Price'].interpolate(method='linear', inplace=True)
        self.dataset['Recommended_Customer_Price'] = pd.qcut(self.dataset['Recommended_Customer_Price'], 5)

        self.dataset['nb_of_Threads'].interpolate(method='linear', inplace=True)
        self.dataset['nb_of_Threads'].fillna(self.dataset['nb_of_Threads'].mode()[0], inplace=True)
        self.dataset['nb_of_Threads'] = self.dataset['nb_of_Threads'].astype(int)

        self.dataset['Processor_Base_Frequency'] = self.dataset['Processor_Base_Frequency'].apply(self.frequency_units)

        self.dataset['Cache_Size'] = self.dataset['Cache'].apply(lambda x: self.cache_type(x)[0])
        self.dataset['Cache_Type'] = self.dataset['Cache'].apply(lambda x: self.cache_type(x)[1])
        self.dataset = self.dataset.drop('Cache', axis=1)
        self.dataset['Cache_Size'].fillna(self.dataset['Cache_Size'].mode()[0], inplace=True)
        self.dataset['Cache_Type'].fillna(self.dataset['Cache_Type'].mode()[0], inplace=True)

        self.dataset['Bus_Type'] = self.dataset['Bus_Speed'].apply(self.bus_type)
        self.dataset['Bus_Type'].fillna(self.dataset['Bus_Type'].mode()[0], inplace=True)
        self.dataset = self.dataset.drop('Bus_Speed', axis=1)

        yes_no_columns = ['Embedded_Options_Available', 'Conflict_Free', 'ECC_Memory_Supported',
                          'Intel_Hyper_Threading_Technology_', 'Intel_Virtualization_Technology_VTx_', 'Intel_64_',
                          'Idle_States', 'Thermal_Monitoring_Technologies', 'Execute_Disable_Bit']
        for column in yes_no_columns:
            self.dataset[column] = self.dataset[column].map({'Yes': 1, 'No': 0})
            self.dataset[column].interpolate(method='linear', inplace=True)
            self.dataset[column].fillna(self.dataset[column].mode()[0], inplace=True)

        self.dataset['TDP'] = self.dataset['TDP'].str.replace('W', '').astype(float)
        self.dataset['TDP'].fillna(self.dataset['TDP'].mean(), inplace=True)
        self.dataset['Max_nb_of_Memory_Channels'].fillna(self.dataset['Max_nb_of_Memory_Channels'].mode()[0],
                                                         inplace=True)

        self.dataset['Max_Memory_Size'] = self.dataset['Max_Memory_Size'].apply(self.memory_size)
        self.dataset['Max_Memory_Size'].fillna(self.dataset['Max_Memory_Size'].mode()[0], inplace=True)

        self.dataset['Max_Memory_Bandwidth'] = self.dataset['Max_Memory_Bandwidth'].str.replace('GB/s', '').astype(
            float)
        self.dataset['Max_Memory_Bandwidth'].fillna(round(self.dataset['Max_Memory_Bandwidth'].mean(), 2), inplace=True)

        self.dataset['PCI_Express_Revision'] = self.dataset['PCI_Express_Revision'].apply(
            lambda x: 3.0 if '3' in str(x) else 2.0 if '2' in str(x) else 1.0 if '1' in str(x) else 0 if 'No' in str(
                x) else x)
        self.dataset['PCI_Express_Revision'].fillna(self.dataset['PCI_Express_Revision'].mode()[0], inplace=True)

        self.dataset['Max_nb_of_PCI_Express_Lanes'].fillna(self.dataset['Max_nb_of_PCI_Express_Lanes'].mode()[0],
                                                           inplace=True)

        self.dataset['T'] = self.dataset['T'].apply(self.extract_temperature)
        self.dataset['T'].fillna(round(self.dataset['T'].mean(), 2), inplace=True)

        # Combine the Instruction Set and 64-bit, then scale the values
        self.dataset['Instruction_Set'] = self.dataset['Instruction_Set'].apply(
            lambda x: 1 if '64' in str(x) else 0 if '32' in str(x) else pd.NA)
        self.dataset['Instruction_Set'].fillna(self.dataset['Instruction_Set'].mode()[0], inplace=True)
        self.dataset['Instruction_Set'] = self.dataset['Instruction_Set'].astype(float) + self.dataset[
            'Intel_64_'].astype(float)
        scaler = MinMaxScaler()
        self.dataset['Instruction_Set'] = scaler.fit_transform(self.dataset[['Instruction_Set']])
        self.dataset = self.dataset.drop('Intel_64_', axis=1)

        # Extract the instruction set extensions and encode them
        instruction_extensions = ['SSE', 'SSE2', 'SSE3', 'SSSE3', 'SSE4', 'AVX', 'MMX', 'AES', 'IMCI', 'AV2']
        for extension in instruction_extensions:
            self.dataset[extension] = self.dataset['Instruction_Set_Extensions'].apply(
                lambda x: 1 if extension in str(x) else 0)
        self.dataset = self.dataset.drop('Instruction_Set_Extensions', axis=1)

        memory_types = ['DDR4', 'DDR3', 'DDR3L', 'DDR2', 'LPDDR3']
        for memory in memory_types:
            self.dataset[memory] = self.dataset['Memory_Types'].apply(lambda x: 1 if memory in str(x) else 0)
        self.dataset = self.dataset.drop('Memory_Types', axis=1)

        # Encode the labels
        label_encoder = LabelEncoder()
        self.dataset['Product_Collection'] = label_encoder.fit_transform(self.dataset['Product_Collection'])
        self.dataset['Lithography'] = label_encoder.fit_transform(self.dataset['Lithography'])
        self.dataset['Cache_Type'] = label_encoder.fit_transform(self.dataset['Cache_Type'])
        self.dataset['Bus_Type'] = label_encoder.fit_transform(self.dataset['Bus_Type'])
        self.dataset['Recommended_Customer_Price'] = label_encoder.fit_transform(
            self.dataset['Recommended_Customer_Price'])

        return self.dataset
