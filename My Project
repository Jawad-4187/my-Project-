import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

def clean_data(df):
    df.columns = df.columns.str.replace(' ', '_')
    df['MS_SubClass'] = df['MS_SubClass'].astype(str)
    df['Yr_Sold'] = df['Yr_Sold'].astype(int)

    missing = df.isnull().sum() / len(df)
    drop_cols = missing[missing > 0.8].index
    df.drop(columns=drop_cols, inplace=True)

    df['Lot_Frontage'] = df['Lot_Frontage'].fillna(df['Lot_Frontage'].median())

    df.drop_duplicates(inplace=True)

    limit = df['SalePrice'].quantile(0.99)
    df['SalePrice'] = np.where(df['SalePrice'] > limit, limit, df['SalePrice'])

    return df

# --- PHASE 1: CLEANING ---

# 1. Load Data
df = pd.read_csv("C:\\Users\\acer21\\Downloads\\AmesHousing.csv")

# 2. Columns
df.columns = df.columns.str.replace(' ', '_')

# 3. Explore
print(df.info())
print(df.head())
print(df.shape)
# 4. Types
df['MS_SubClass'] = df['MS_SubClass'].astype(str)
df['Yr_Sold'] = df['Yr_Sold'].astype(int)

# 5. Missing Values
missing = df.isnull().sum() / len(df)
drop_cols = missing[missing > 0.8].index
df.drop(columns=drop_cols, inplace=True)

df['Lot_Frontage'] = df['Lot_Frontage'].fillna(df['Lot_Frontage'].median())

# 6. Duplicates
print("Duplicates:", df.duplicated().sum())
df.drop_duplicates(inplace=True)

# 7. Outliers
limit = df['SalePrice'].quantile(0.99)
df['SalePrice'] = np.where(df['SalePrice'] > limit, limit, df['SalePrice'])

sns.boxplot(x=df['SalePrice'])
plt.title("SalePrice Before/After Capping")
plt.show()

print("Nulls in SalePrice:", df['SalePrice'].isnull().sum())
print("All prices > 0:", (df['SalePrice'] > 0).all())
print("Shape:", df.shape)

# 8. Save Cleaned
df.to_csv('AmesHousing_cleaned.csv', index=False)

df.to_csv('data/cleaned/AmesHousing_cleaned.csv', index=False)

# --- PHASE 2: FEATURES ---
df_eda = df.copy()

# 1. Encoding
df = pd.get_dummies(df, columns=['Neighborhood'], drop_first=True)
df = pd.get_dummies(df, columns=['Bldg_Type'], drop_first=True)

qual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
df['Exter_Qual_Num'] = df['Exter_Qual'].map(qual_map).fillna(3)

# 2. Scaling t
scaler = StandardScaler()
df['Gr_Liv_Area_Scaled'] = scaler.fit_transform(df[['Gr_Liv_Area']])
df['Lot_Area_Scaled'] = scaler.fit_transform(df[['Lot_Area']])

# 3. New Features 3
df['Price_Per_Sqft'] = df['SalePrice'] / np.where(df['Gr_Liv_Area'] == 0, 1, df['Gr_Liv_Area'])
df['House_Age'] = df['Yr_Sold'] - df['Year_Built']
df['Age_Group'] = pd.cut(df['House_Age'],
                         bins=[0, 10, 30, 100],
                         labels=['New', 'Recent', 'Old'])

# 4. Interaction tf
df['Quality_x_Area'] = df['Overall_Qual'] * df['Gr_Liv_Area']

# 5. Log Transform
plt.figure(figsize=(10, 4))
sns.histplot(df['SalePrice'], bins=40)
plt.title("Before Log Transform")
plt.show()

df['Log_SalePrice'] = np.log1p(df['SalePrice'])

plt.figure(figsize=(10, 4))
sns.histplot(df['Log_SalePrice'], bins=40)
plt.title("After Log Transform")
plt.show()

# 6. Save Features
corr_matrix = df.select_dtypes(include=[np.number]).corr().abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

df.drop(columns=to_drop, inplace=True)

df.to_csv('data/cleaned/AmesHousing_features.csv', index=False)

# PHASE 3: EDA (Visuals)

# 1. Price Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['SalePrice'], kde=True, color='blue')
plt.title('House Price Distribution')
plt.show()

# 2. Area vs Price
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Gr_Liv_Area', y='SalePrice', data=df, alpha=0.5)
plt.title('Living Area vs Sale Price')
plt.show()

# 3. Correlations (Top 10)
plt.figure(figsize=(12, 8))
top_corr = df.select_dtypes(include=[np.number]).corr()['SalePrice'].sort_values(ascending=False).head(10)
top_corr.plot(kind='barh', color='green')
plt.title('Top 10 Features Correlated with Price')
plt.show()

# 4.  Grouped Boxplot (Outliers Visualization)
plt.figure(figsize=(10, 6))
sns.boxplot(x='Age_Group', y='SalePrice', data=df, hue='Age_Group', palette='Set2', legend=False)
plt.title('SalePrice Comparison by Age Group')
plt.show()

# 5. KDE Comparison (Distribution Shape)
plt.figure(figsize=(10, 6))
sns.kdeplot(df['SalePrice'], fill=True, color='purple')
plt.title('KDE of SalePrice')
plt.show()

# 6. Grouped Analysis (Neighborhood vs Price)
plt.figure(figsize=(12, 6))
df_eda.groupby('Neighborhood')['SalePrice'].mean().sort_values(ascending=False).head(10).plot(kind='bar', color='red')
plt.title('Top Neighborhoods by Average Price')
plt.xticks(rotation=45)
plt.show()

# --- PHASE 4: MATH BASICS (MANUAL CALCULATIONS) ---

# 1. Manual Mean & Std (Using NumPy as required)
prices = df['SalePrice'].values
m_mean = np.sum(prices) / len(prices)
m_std = np.sqrt(np.sum((prices - m_mean)**2) / len(prices))
print(f"Manual Mean: {m_mean:.2f}")
print(f"Manual Std Dev: {m_std:.2f}")

# 2. Manual Standardization (Broadcasting)
# Formula: z = (X - mean) / std
area_values = df['Gr_Liv_Area'].values
z_manual = (area_values - np.mean(area_values)) / np.std(area_values)
print(f"First 5 Manual Z-scores: {z_manual[:5]}")

# 3. Probability Estimation
# Probability of SalePrice > 250k given Overall_Qual >= 8
high_qual_houses = df[df['Overall_Qual'] >= 8]
prob_estimate = (high_qual_houses['SalePrice'] > 250000).mean()
print(f"Probability (Price > 250k | High Quality): {prob_estimate:.2%}")

# 4. Cosine Similarity (Highest vs Lowest value records)
from sklearn.metrics.pairwise import cosine_similarity

numeric_cols = df.select_dtypes(include=[np.number]).columns
highest_val_record = df.loc[df['SalePrice'].idxmax(), numeric_cols].values.reshape(1, -1)
lowest_val_record = df.loc[df['SalePrice'].idxmin(), numeric_cols].values.reshape(1, -1)

cos_sim = cosine_similarity(highest_val_record, lowest_val_record)
print(f"Cosine Similarity (Max vs Min Home): {cos_sim[0][0]:.4f}")

# --- BONUS DEEP DIVE ---

# 1. Manual Stats
prices = df['SalePrice'].values
m_mean = np.sum(prices) / len(prices)
m_std = np.sqrt(np.sum((prices - m_mean)**2) / len(prices))
print(f"Manual Mean: {m_mean:.2f}")
print(f"Manual Std Dev: {m_std:.2f}")

# 2. Manual Standardization
area_values = df['Gr_Liv_Area'].values
z_manual = (area_values - np.mean(area_values)) / np.std(area_values)
print(f"First 5 Manual Z-scores: {z_manual[:5]}")

# 3. Probability Estimation
high_qual_houses = df[df['Overall_Qual'] >= 8]
prob_estimate = (high_qual_houses['SalePrice'] > 250000).mean()
print(f"Probability (Price > 250k | High Quality): {prob_estimate:.2%}")

# 4. Cosine Similarity
numeric_cols = df.select_dtypes(include=[np.number]).columns
highest_val_record = df.loc[df['SalePrice'].idxmax(), numeric_cols].values.reshape(1, -1)
lowest_val_record = df.loc[df['SalePrice'].idxmin(), numeric_cols].values.reshape(1, -1)
cos_sim = cosine_similarity(highest_val_record, lowest_val_record)
print(f"Cosine Similarity (Max vs Min Home): {cos_sim[0][0]:.4f}")

# 5. BONUS: Feature Predictive Power Analysis
print("\n--- FEATURE PREDICTIVE POWER ANALYSIS ---")
X_area = df[['Gr_Liv_Area']]
X_quality = df[['Overall_Qual']]
X_interaction = df[['Quality_x_Area']]
y = df['SalePrice']

r2_area = LinearRegression().fit(X_area, y).score(X_area, y)
r2_qual = LinearRegression().fit(X_quality, y).score(X_quality, y)
r2_inter = LinearRegression().fit(X_interaction, y).score(X_interaction, y)

print(f"R-Squared (Area alone): {r2_area:.4f}")
print(f"R-Squared (Quality alone): {r2_qual:.4f}")
print(f"R-Squared (Quality x Area Interaction): {r2_inter:.4f}")

print("All Phases and Bonus Tasks Completed Successfully")
