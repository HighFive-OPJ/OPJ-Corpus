# OPJ-Corpus
Group members: Ivana Kollert, Lorena Mitrović, Marija Nadoveza, Ana Sabo, Mia Sambolec.
<p>Information about the project so far:</p>
<p>We have collected croatian movie reviews of the newest thriller, sci-fy and horror movies.</p>
<p>We have collected about 3000 sentences and organized them by movie, review and sentence ID.</p>
<p>We have made a pilot annotation campaign with 150 sentences. All of us were annotators in this pilot campaign. We used 0 for positive sentiment, 1 for neutral sentiment and 2 for negative sentiment. After running the code, we got the agreement rate of 0.3182 and this is a Fair agreement.</p>
<p>We have made final two annotation campaigns. In the first round, we made an annotation with 3 annotators and in the final round we made a single annotation for all the 3259 sentences.</p>
<p>We used a total of five sentiments for the two final annotations. We used 1 for negative, 2 for neutral, 3 for positive, 4 for mixed and 5 for sarcasm.</p>
<p>We have created a new code that would help us to calculate the inter-rater agreement more easily and effectively using fleiss Kappa.</p>
<p>The inter-rater agreement we got in the end is 0.7961 and that is a substantial agreement.</p>
<p>We have created a new CSV file that contains only the two colums of our corpus labeled "sentence" and "label". The name of this file is Movies - Final Annotation - test.</p>
<p>We have used a code to split our CSV file into two parts, a Train set and a Test set. The Train set includes 75 to 80% of the original file and the Test set includes 25 to 30% of the original CSV files. Both the Train and Test set have been saved as two separates CSV files.</p>
<p>We conducted a test on the Test set in which we calculated the average number of words per sentence, the largest and the smalest sentece. We also calculated the number of sentences per label in the Test set.</p>
<p>The Test set statistics is as follows: </p>
<p> - 653 total sentences</p>
<p>Label Distribution: </p>
<p>     - Label 1: 58</p>
<p>     - Label 2: 379</p>
<p>     - Label 3: 165</p>
<p>     - Label 4: 49</p>
<p>     - Label 5: 2</p>
<p>Sentence Lenght (in words):</p>
<p>     - Average: 21.33</p>
<p>     - Shortest: 1</p>
<p>     - Longest: 95</p>
<p>The results are saved in the file "Dataset.md" See link below. </p>
<p>https://github.com/HighFive-OPJ/OPJ-Corpus/tree/main/Exploratory%20data%20analysis</p>
<p> We implemented our data with machine learning. We used SVM and KNN algorithms. </p>
<p> These are the results:</p>
<p> Train1: </p>
<p>    - Test 1:</p>
<p>      - Precision: 0.5315</p>
<p>      - Recall: 0.5402</p>
<p>      - F1-Score: 0.5353</p>
<p>      - Accuracy: 0.5402</p>
<p>    - Test 3:</p>
<p>      - Precision: 0.5301</p>
<p>      - Recall: 0.5320</p>
<p>      - F1-Score: 0.5303</p>
<p>      - Accuracy: 0.5320</p>
<p> Train3: </p>
<p>    - Test 1:</p>
<p>      - Precision: 0.3086</p>
<p>      - Recall: 0.5556</p>
<p>      - F1-Score: 0.3968</p>
<p>      - Accuracy: 0.5556</p>
<p>    - Test 3:</p>
<p>      - Precision: 0.4862</p>
<p>      - Recall: 0.4912</p>
<p>      - F1-Score: 0.3570</p>
<p>      - Accuracy: 0.4912</p>
