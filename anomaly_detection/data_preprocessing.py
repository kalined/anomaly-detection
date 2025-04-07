import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./data/csv_files/combined_with_labels_and_synthetic.csv")

print(df.head(10))

print("Describe a dataframe:", df.describe())
print("Dataframe shape:", df.shape)

df_events = df["Label"]
print(df_events.describe())

print(df_events.unique())

print(df_events.value_counts())

count_events = df_events.value_counts()

# Plot value counts

count_events.plot(kind='bar', color='blue')
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('value Counts of Events')
plt.show()