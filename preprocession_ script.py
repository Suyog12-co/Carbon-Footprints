import pandas as pd
import numpy as np
import ast

df = pd.read_csv('dataset.csv')  # adjust path

# --- 1. Ordinal encodings ---
shower_map = {'less frequently': 0, 'daily': 1, 'more frequently': 2, 'twice a day': 3}
df['How Often Shower'] = df['How Often Shower'].map(shower_map)

energy_eff_map = {'No': 0, 'Sometimes': 1, 'Yes': 2}
df['Energy efficiency'] = df['Energy efficiency'].map(energy_eff_map)

air_travel_map = {'never': 0, 'rarely': 1, 'frequently': 2, 'very frequently': 3}
df['Frequency of Traveling by Air'] = df['Frequency of Traveling by Air'].map(air_travel_map)

# --- 2. Binary encoding ---
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# --- 3. Vehicle Type: fill structural NaN, then one-hot ---
df['Vehicle Type'] = df['Vehicle Type'].fillna('none')

# --- 4. Multi-hot encoding for Recycling ---
def parse_list_str(s):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []

df['Recycling_parsed'] = df['Recycling'].apply(parse_list_str)
for item in ['Paper', 'Plastic', 'Glass', 'Metal']:
    df[f'Recycles_{item}'] = df['Recycling_parsed'].apply(lambda x: 1 if item in x else 0)

# --- 5. Multi-hot encoding for Cooking_With ---
df['Cooking_parsed'] = df['Cooking_With'].apply(parse_list_str)
for item in ['Stove', 'Oven', 'Microwave', 'Grill', 'Airfryer']:
    df[f'Cooking_{item}'] = df['Cooking_parsed'].apply(lambda x: 1 if item in x else 0)

# --- 6. Drop original list/raw columns now replaced ---
df = df.drop(columns=['Recycling', 'Cooking_With', 'Recycling_parsed', 'Cooking_parsed'])

# --- 7. One-hot encode remaining nominal categoricals ---
nominal_cols = ['Body Type', 'Diet', 'Heating Energy Source', 'Transport',
                 'Vehicle Type', 'Social Activity', 'Waste Bag Size']
df = pd.get_dummies(df, columns=nominal_cols, drop_first=False)

# --- 8. Log-transform target ---
df['CarbonEmission_log'] = np.log1p(df['CarbonEmission'])

# --- 9. Sanity checks ---
print(df.shape)
print(df.isnull().sum().sum())  # should be 0
print(df.dtypes.value_counts())

df.to_csv('cleaned_data.csv', index=False)