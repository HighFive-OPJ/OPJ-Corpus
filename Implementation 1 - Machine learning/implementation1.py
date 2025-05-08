import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

file_path = 'Train-1.tsv'
data = pd.read_csv(file_path, sep='\t')

X = data['Sentence']  
y = data['Label']     

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)

plt.figure(figsize=(8, 6))
y.value_counts().plot(kind='bar', color='skyblue')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.show()

svm_model = SVC(kernel='linear', random_state=42, class_weight='balanced')
svm_model.fit(X_train, y_train)

svm_predictions = svm_model.predict(X_test)

svm_precision = precision_score(y_test, svm_predictions, average='weighted', zero_division=0)
svm_recall = recall_score(y_test, svm_predictions, average='weighted', zero_division=0)
svm_f1 = f1_score(y_test, svm_predictions, average='weighted', zero_division=0)
svm_accuracy = accuracy_score(y_test, svm_predictions)

print("SVM Model Performance:")
print(f"Precision: {svm_precision:.4f}")
print(f"Recall: {svm_recall:.4f}")
print(f"F1-Score: {svm_f1:.4f}")
print(f"Accuracy: {svm_accuracy:.4f}")

knn_model = KNeighborsClassifier(n_neighbors=10)
knn_model.fit(X_train, y_train)

knn_predictions = knn_model.predict(X_test)

knn_precision = precision_score(y_test, knn_predictions, average='weighted', zero_division=0)
knn_recall = recall_score(y_test, knn_predictions, average='weighted', zero_division=0)
knn_f1 = f1_score(y_test, knn_predictions, average='weighted', zero_division=0)
knn_accuracy = accuracy_score(y_test, knn_predictions)

print("\nKNN Model Performance:")
print(f"Precision: {knn_precision:.4f}")
print(f"Recall: {knn_recall:.4f}")
print(f"F1-Score: {knn_f1:.4f}")
print(f"Accuracy: {knn_accuracy:.4f}")

print("\nConfusion Matrix for SVM Model:")
svm_cm = confusion_matrix(y_test, svm_predictions)
print(svm_cm)

print("\nConfusion Matrix for KNN Model:")
knn_cm = confusion_matrix(y_test, knn_predictions)
print(knn_cm)
