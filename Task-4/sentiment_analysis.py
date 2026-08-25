import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
columns = ["ID", "Entity", "Sentiment", "Tweet"]

train_data = pd.read_csv("twitter_training.csv", header=None, names=columns)
validation_data = pd.read_csv("twitter_validation.csv", header=None, names=columns)

# Combine both datasets
data = pd.concat([train_data, validation_data], ignore_index=True)

# Remove missing values
data.dropna(subset=["Entity", "Sentiment", "Tweet"], inplace=True)

# Display basic information
print("Twitter Sentiment Analysis")
print("--------------------------")
print("Total records:", len(data))

print("\nSentiment Distribution:")
print(data["Sentiment"].value_counts())

# Plot overall sentiment distribution
plt.figure(figsize=(8, 5))
data["Sentiment"].value_counts().plot(kind="bar")
plt.title("Overall Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Tweets")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Sentiment by entity
entity_sentiment = pd.crosstab(data["Entity"], data["Sentiment"])

print("\nSentiment by Entity:")
print(entity_sentiment)

# Plot sentiment patterns for top 10 entities
top_entities = data["Entity"].value_counts().head(10).index
top_data = data[data["Entity"].isin(top_entities)]

plt.figure(figsize=(12, 6))
pd.crosstab(top_data["Entity"], top_data["Sentiment"]).plot(
    kind="bar", figsize=(12, 6)
)
plt.title("Sentiment Patterns Across Top 10 Entities")
plt.xlabel("Entity")
plt.ylabel("Number of Tweets")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()