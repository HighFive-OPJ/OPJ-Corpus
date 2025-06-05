import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

df = pd.read_csv("Movies - Final Annotation - test.csv")  
df.dropna(subset=["sentence", "label"], inplace=True)

corpus_size = len(df)
test_ratio = 0.25 if 300 <= int(corpus_size * 0.25) <= 500 else 0.20
print(f"Corpus size: {corpus_size} — Using test ratio: {test_ratio}")

train_df, test_df = train_test_split(df, test_size=test_ratio, random_state=42)

train_df.to_csv("Train_HighFive.csv", index=False, encoding="utf-8")
test_df.to_csv("Test_HighFive.csv", index=False, encoding="utf-8")

label_counts = Counter(test_df["label"])
sentence_lengths = test_df["sentence"].apply(lambda s: len(str(s).split()))

report = "# Test Set Statistics\n\n"
report += f"Total test sentences: {len(test_df)}\n\n"

report += "## Label Distribution\n"
for label in range(0, 3):
    report += f"- Label {label}: {label_counts.get(label, 0)}\n"

report += "\n## Sentence Length (in words)\n"
report += f"- Average: {sentence_lengths.mean():.2f}\n"
report += f"- Shortest: {sentence_lengths.min()}\n"
report += f"- Longest: {sentence_lengths.max()}\n"

with open("Dataset.md", "w", encoding="utf-8") as f:
    f.write(report)

print("Split complete:")
print("- Train set saved to 'train_set.csv'")
print("- Test set saved to 'test_set.csv'")
print("- Test statistics saved to 'dataset.md'")
