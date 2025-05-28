# OPJ-Corpus
Group members: Ivana Kollert, Lorena Mitrović, Marija Nadoveza, Ana Sabo, Mia Sambolec.
<p>Information about the project so far:</p>
<p>We have collected croatian movie reviews of the newest thriller, sci-fy and horror movies.</p>
<p>We have collected about 3000 sentences and organized them by movie, review and sentence ID.</p>
<p>We have made a pilot annotation campaign with 150 sentences. All of us were annotators in this pilot campaign. We used 0 for positive sentiment, 1 for neutral sentiment and 2 for negative sentiment. After running the code, we got the agreement rate of 0.3182 and this is a Fair agreement.</p>
<p>We have made final two annotation campaigns. In the first round, we made an annotation with 3 annotators and in the final round we made a single annotation for all the 3259 sentences.</p>
<p>We used a total of three sentiments for the two final annotations. We used 0 for positive, 1 for negative and 2 for neutral sentiment.</p>
<p>We have created a new code that would help us to calculate the inter-rater agreement more easily and effectively using fleiss Kappa.</p>
<p>The inter-rater agreement we got in the end is 0.7869 and that is a substantial agreement.</p>
<p>We have created a new CSV file that contains only the two colums of our corpus labeled "sentence" and "label". The name of this file is Movies - Final Annotation - test.</p>
<p>We have used a code to split our CSV file into two parts, a Train set and a Test set. The Train set includes 75 to 80% of the original file and the Test set includes 25 to 30% of the original CSV files. Both the Train and Test set have been saved as two separates CSV files.</p>
<p>We conducted a test on the Test set in which we calculated the average number of words per sentence, the largest and the smalest sentece. We also calculated the number of sentences per label in the Test set.</p>
<p>The Test set statistics is as follows: </p>
<p> - 653 total sentences</p>

| Category             | Metric     | Value |
|----------------------|------------|-------|
| **Label Distribution** | Label 0    | 165   |
|                      | Label 1    | 58    |
|                      | Label 2    | 430   |
| **Sentence Length**    | Average    | 21.33 |
|                      | Shortest   | 1     |
|                      | Longest    | 95    |

<p>The results are saved in the file "Dataset.md" See link below. </p>
<p>https://github.com/HighFive-OPJ/OPJ-Corpus/blob/9415303240584f8f93e58f7c4a190b8b235388f2/Exploratory%20data%20analysis/Dataset.md</p>
<p> We implemented our data with machine learning. We used SVM and KNN algorithms. </p>
<p> These are the results:</p>

| #      | Method                     | Algorithm                | Train                                | Test 1: Group 1 (ours)                                             | Test 2: Group 2                                               | Test 3: Group 3                                               |
|--------|----------------------------|--------------------------|--------------------------------------|---------------------------------------------------------------|---------------------------------------------------------------|---------------------------------------------------------------|
| 1.a.i  | Machine learning (2 methods) | SVM                      | Train 1 or 2 or 3 / [respective own] | Precision: 0.6301, Recall: 0.6684, F1: 0.5532, Accuracy: 0.6684 | Precision: 0.5402, Recall: 0.6099, F1: 0.4935, Accuracy: 0.6099 | Precision: 0.5677, Recall: 0.5714, F1: 0.5661, Accuracy: 0.5714 |
| 1.a.ii |                            | SVM                      | TRAIN                                | Precision: 0.6119, Recall: 0.6782, F1: 0.6182, Accuracy: 0.6782 | Precision: 0.5699, Recall: 0.6222, F1: 0.5624, Accuracy: 0.6222 | Precision: 0.6070, Recall: 0.6073, F1: 0.6055, Accuracy: 0.6073 |
| 1.b.i  |                            | K-Nearest Neighbors (KNN) | Train 1 or 2 or 3 / [respective own] | Precision: 0.5941, Recall: 0.6684, F1: 0.5544, Accuracy: 0.6684 | Precision: 0.4766, Recall: 0.5964, F1: 0.4870, Accuracy: 0.5964 | Precision: 0.5125, Recall: 0.5126, F1: 0.5124, Accuracy: 0.5126 |
| 1.b.ii |                            | KNN                      | TRAIN                                | Precision: 0.5066, Recall: 0.6398, F1: 0.5233, Accuracy: 0.6398 | Precision: 0.5641, Recall: 0.6117, F1: 0.5479, Accuracy: 0.6117 | Precision: 0.5460, Recall: 0.5454, F1: 0.5451, Accuracy: 0.5454 |

<p>The best results for SVM shows Test-1 for all categories. The best results in the TRAIN category shows Train-1</p>
<p>The best results for KNN shows Test-1, but Test-3 shows also good results. In the TRAIN category Train-1 shows best results in Recall and Accuracy and Train-2 shows best results in Precision and the F1-Score.</p>

**Deep Learning**

<p> We conducted deep learning using two algorithms - LSTM and CNN.</p>
<p> We trained both models on the same training dataset and tested them on the test files. </p>
<p> These are the results:</p>

| #     | Method                 | Algorithm | Test 1 (ours)                                               | Test 2                                                  | Test 3                                                  |
|-------|------------------------|-----------|---------------------------------------------------------|---------------------------------------------------------|---------------------------------------------------------|
| 1.a   | Deep learning (2 methods) | LSTM      | Precision: 0.4114<br>Recall: 0.4006<br>F1-Score: 0.3621<br>Accuracy: 0.3982 | Precision: 0.5091<br>Recall: 0.4911<br>F1-Score: 0.4757<br>Accuracy: 0.5479 | Precision: 0.7101<br>Recall: 0.6905<br>F1-Score: 0.6899<br>Accuracy: 0.6898 |
| 1.b   |                        | CNN model | Precision: 0.4023<br>Recall: 0.3577<br>F1-Score: 0.3388<br>Accuracy: 0.3675 | Precision: 0.5364<br>Recall: 0.5324<br>F1-Score: 0.5165<br>Accuracy: 0.6046 | Precision: 0.7824<br>Recall: 0.7776<br>F1-Score: 0.7738<br>Accuracy: 0.7768 |

<p> Test 3 shows the best results for both LSTM and CNN model for all categories.</p>

<p> When comparing the results with machine learning, SVM TRAIN offered the best F1-score and accuracy for Test 1 and Test 2 and showed the most consistent performance across all test group.</p>
<p> LSTM offered good results in Test 3 when compared to other tests, where it underperformed. </p>
<p> For Test 3, CNN offered the best results for all categories. </p>
<p> KNN	showed moderate results throughout the tests, but never achieved the best results. </p>
