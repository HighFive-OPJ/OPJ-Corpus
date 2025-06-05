import pandas as pd
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

file_path = 'Test-3.tsv'
data = pd.read_csv(file_path, sep="\t", names=["Sentence", "Label"], skiprows=1, quoting=csv.QUOTE_NONE, encoding="utf-8")
data.columns = data.columns.str.strip()

data = data.dropna(subset=['Sentence', 'Label'])
data['Sentence'] = data['Sentence'].astype(str)

X = data['Sentence']
y = data['Label']

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.3, random_state=42, stratify=y)

plt.figure(figsize=(8, 6))
y.value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('class_distribution.png')
plt.close()

svm_model = SVC(kernel='rbf', degree=3, random_state=42, class_weight='balanced')
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)

print("SVM Model Performance:")
print(f"Precision: {precision_score(y_test, svm_predictions, average='weighted', zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, svm_predictions, average='weighted', zero_division=0):.4f}")
print(f"F1-Score:  {f1_score(y_test, svm_predictions, average='weighted', zero_division=0):.4f}")
print(f"Accuracy:  {accuracy_score(y_test, svm_predictions):.4f}")

param_grid = {'n_neighbors': list(range(3, 21, 2))}
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_knn = grid_search.best_estimator_
knn_predictions = best_knn.predict(X_test)

print("\nKNN Model Performance:")
print(f"Best k:    {grid_search.best_params_['n_neighbors']}")
print(f"Precision: {precision_score(y_test, knn_predictions, average='weighted', zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, knn_predictions, average='weighted', zero_division=0):.4f}")
print(f"F1-Score:  {f1_score(y_test, knn_predictions, average='weighted', zero_division=0):.4f}")
print(f"Accuracy:  {accuracy_score(y_test, knn_predictions):.4f}")

def plot_conf_matrix(cm, title, filename):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = range(len(cm))
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                     ha='center', va='center',
                     color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

plot_conf_matrix(confusion_matrix(y_test, svm_predictions), 'SVM Confusion Matrix', 'svm_conf_matrix.png')
plot_conf_matrix(confusion_matrix(y_test, knn_predictions), 'KNN Confusion Matrix', 'knn_conf_matrix.png')
