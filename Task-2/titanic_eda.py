# Internship T4 - Task 2
# Data Cleaning and Exploratory Data Analysis (EDA)
# Titanic Dataset

import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("Titanic-Dataset.csv")

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATASET SHAPE ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

# ==========================================
# 2. BASIC INFORMATION
# ==========================================

print("\n========== DATA INFORMATION ==========")
print(df.info())

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe(include="all"))

# ==========================================
# 3. CHECK MISSING VALUES
# ==========================================

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== MISSING VALUE PERCENTAGE ==========")
missing_percentage = (df.isnull().sum() / len(df)) * 100
print(missing_percentage.round(2))

# ==========================================
# 4. DATA CLEANING
# ==========================================

# Fill missing Age values with median
if "Age" in df.columns:
    df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked values with mode
if "Embarked" in df.columns:
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Fill missing Fare values with median
if "Fare" in df.columns:
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# Remove Cabin because it contains many missing values
if "Cabin" in df.columns:
    df = df.drop(columns=["Cabin"])

# Remove duplicate rows
df = df.drop_duplicates()

print("\n========== AFTER DATA CLEANING ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nRemaining missing values:")
print(df.isnull().sum())

# ==========================================
# 5. SURVIVAL ANALYSIS
# ==========================================

print("\n========== SURVIVAL COUNT ==========")
print(df["Survived"].value_counts())

survival_rate = df["Survived"].mean() * 100
print(f"\nOverall Survival Rate: {survival_rate:.2f}%")

# ==========================================
# 6. SURVIVAL BY GENDER
# ==========================================

if "Sex" in df.columns:

    print("\n========== SURVIVAL BY GENDER ==========")

    gender_survival = df.groupby("Sex")["Survived"].mean() * 100
    print(gender_survival.round(2))

    plt.figure(figsize=(8, 5))

    gender_survival.plot(kind="bar")

    plt.title("Survival Rate by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Survival Rate (%)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

# ==========================================
# 7. SURVIVAL BY PASSENGER CLASS
# ==========================================

if "Pclass" in df.columns:

    print("\n========== SURVIVAL BY PASSENGER CLASS ==========")

    class_survival = df.groupby("Pclass")["Survived"].mean() * 100
    print(class_survival.round(2))

    plt.figure(figsize=(8, 5))

    class_survival.plot(kind="bar")

    plt.title("Survival Rate by Passenger Class")
    plt.xlabel("Passenger Class")
    plt.ylabel("Survival Rate (%)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

# ==========================================
# 8. AGE DISTRIBUTION
# ==========================================

if "Age" in df.columns:

    plt.figure(figsize=(9, 5))

    plt.hist(df["Age"], bins=20, edgecolor="black")

    plt.title("Age Distribution of Titanic Passengers")
    plt.xlabel("Age")
    plt.ylabel("Number of Passengers")
    plt.tight_layout()
    plt.show()

# ==========================================
# 9. FARE DISTRIBUTION
# ==========================================

if "Fare" in df.columns:

    plt.figure(figsize=(9, 5))

    plt.hist(df["Fare"], bins=20, edgecolor="black")

    plt.title("Fare Distribution")
    plt.xlabel("Fare")
    plt.ylabel("Number of Passengers")
    plt.tight_layout()
    plt.show()

# ==========================================
# 10. AGE VS SURVIVAL
# ==========================================

if "Age" in df.columns:

    plt.figure(figsize=(8, 5))

    df.boxplot(column="Age", by="Survived")

    plt.title("Age Distribution by Survival")
    plt.suptitle("")
    plt.xlabel("Survived (0 = No, 1 = Yes)")
    plt.ylabel("Age")
    plt.tight_layout()
    plt.show()

# ==========================================
# 11. CORRELATION ANALYSIS
# ==========================================

numeric_columns = df.select_dtypes(include="number")

print("\n========== CORRELATION MATRIX ==========")
print(numeric_columns.corr().round(2))

plt.figure(figsize=(10, 7))

plt.imshow(
    numeric_columns.corr(),
    aspect="auto"
)

plt.colorbar()

plt.xticks(
    range(len(numeric_columns.columns)),
    numeric_columns.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(numeric_columns.columns)),
    numeric_columns.columns
)

plt.title("Correlation Matrix")

plt.tight_layout()
plt.show()

# ==========================================
# 12. KEY EDA FINDINGS
# ==========================================

print("\n========== KEY EDA FINDINGS ==========")

if "Sex" in df.columns:
    female_rate = df.loc[df["Sex"] == "female", "Survived"].mean() * 100
    male_rate = df.loc[df["Sex"] == "male", "Survived"].mean() * 100

    print(f"Female survival rate: {female_rate:.2f}%")
    print(f"Male survival rate: {male_rate:.2f}%")

if "Pclass" in df.columns:
    print("\nSurvival rate by passenger class:")
    print(class_survival.round(2))

print("\nEDA completed successfully!")