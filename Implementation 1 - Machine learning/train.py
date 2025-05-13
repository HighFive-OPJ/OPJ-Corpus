import csv

filenames = ['Train-1.tsv', 'Train-2.tsv', 'Train-3.tsv']

TRAIN = set()

for filename in filenames:
    with open(filename, 'r', encoding='utf-8') as file:
        tsv_reader = csv.reader(file, delimiter='\t')
        header = next(tsv_reader)
        try:
            sentence_idx = header.index("Sentence")
            label_idx = header.index("Label")
        except ValueError:
            raise ValueError(f"Required columns not found in {filename}")

        for row in tsv_reader:
            if len(row) > max(sentence_idx, label_idx):  
                TRAIN.add((row[sentence_idx], row[label_idx]))

with open('Combined_Train.tsv', 'w', encoding='utf-8', newline='') as outfile:
    tsv_writer = csv.writer(outfile, delimiter='\t')
    tsv_writer.writerow(["Sentence", "Label"])  
    for row in TRAIN:
        tsv_writer.writerow(row)

print(f"Combined TRAIN set written to 'Combined_Train.tsv' with {len(TRAIN)} unique rows.")
