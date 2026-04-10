import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create folder for saving plots
os.makedirs("screenshots", exist_ok=True)

# Load Titanic dataset
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSummary Statistics:")
print(df.describe())

print("\nMedian Values:")
print(df.median(numeric_only=True))

print("\nMode Values:")
print(df.mode().iloc[0])

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Histograms
df[numeric_cols].hist(figsize=(12, 10), bins=20, edgecolor='black')
plt.suptitle("Histograms of Numeric Features")
plt.tight_layout()
plt.savefig("screenshots/histograms.png")
plt.show()

# Boxplots
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.savefig(f"screenshots/boxplot_{col}.png")
    plt.show()

# Survival count
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.savefig("screenshots/survival_count.png")
plt.show()

# Passenger class count
plt.figure(figsize=(6, 4))
sns.countplot(x='Pclass', data=df)
plt.title("Passenger Class Count")
plt.savefig("screenshots/pclass_count.png")
plt.show()

# Survival by gender
plt.figure(figsize=(6, 4))
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival by Gender")
plt.savefig("screenshots/survival_by_gender.png")
plt.show()

# Survival by class
plt.figure(figsize=(6, 4))
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Survival by Passenger Class")
plt.savefig("screenshots/survival_by_class.png")
plt.show()

# Pairplot
pairplot_data = df[['Survived', 'Pclass', 'Age', 'Fare']].dropna()
pair_plot = sns.pairplot(pairplot_data, hue='Survived')
pair_plot.savefig("screenshots/pairplot.png")
plt.show()

# Correlation matrix
plt.figure(figsize=(8, 6))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.savefig("screenshots/correlation_matrix.png")
plt.show()

# Skewness
print("\nSkewness:")
print(df[numeric_cols].skew())

print("\nInference:")
print("The Titanic dataset contains missing values in Age, Cabin, and Embarked.")
print("The Fare feature has many outliers.")
print("Female passengers had a higher survival rate compared to males.")
print("Passengers in first class had better survival chances than those in lower classes.")
print("Correlation and pairplots help understand feature relationships.")
