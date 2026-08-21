# Internship T4 - Task 1
# Age Distribution Visualization

import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------
# 1. Load the population dataset
# -----------------------------------------

data = pd.read_csv("population_data.csv")


# -----------------------------------------
# 2. Display the dataset
# -----------------------------------------

print("\n========== POPULATION DATA ==========\n")

print(data)


# -----------------------------------------
# 3. Check for missing values
# -----------------------------------------

print("\n========== MISSING VALUES ==========\n")

print(data.isnull().sum())


# -----------------------------------------
# 4. Basic age statistics
# -----------------------------------------

print("\n========== AGE STATISTICS ==========\n")

print(data["Age"].describe())


# -----------------------------------------
# 5. Calculate mean and median age
# -----------------------------------------

mean_age = data["Age"].mean()

median_age = data["Age"].median()

print(f"Mean Age: {mean_age:.2f}")

print(f"Median Age: {median_age:.2f}")


# -----------------------------------------
# 6. Create histogram
# -----------------------------------------

plt.figure(figsize=(10, 6))

plt.hist(
    data["Age"],
    bins=10,
    edgecolor="black"
)


# -----------------------------------------
# 7. Add title and labels
# -----------------------------------------

plt.title(
    "Distribution of Age in the Population",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel(
    "Age",
    fontsize=12
)

plt.ylabel(
    "Number of People",
    fontsize=12
)


# -----------------------------------------
# 8. Add mean and median lines
# -----------------------------------------

plt.axvline(
    mean_age,
    linestyle="--",
    linewidth=2,
    label=f"Mean Age: {mean_age:.2f}"
)

plt.axvline(
    median_age,
    linestyle=":",
    linewidth=2,
    label=f"Median Age: {median_age:.2f}"
)


# -----------------------------------------
# 9. Add legend and grid
# -----------------------------------------

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)


# -----------------------------------------
# 10. Display the chart
# -----------------------------------------

plt.tight_layout()

plt.show()