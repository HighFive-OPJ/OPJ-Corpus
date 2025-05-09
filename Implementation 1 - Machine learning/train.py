import csv

# Define the filenames
filenames = ['Train-1.tsv', 'Train-2.tsv', 'Train-3.tsv']

# Initialize the TRAIN set
TRAIN = set()

# Read and combine all train files
for filename in filenames:
    with open(filename, 'r', encoding='utf-8') as file:
        tsv_reader = csv.reader(file, delimiter='\t')
        for row in tsv_reader:
            TRAIN.add(tuple(row))  # Convert list to tuple for set compatibility

# Write combined data to a new .tsv file
with open('Combined_Train.tsv', 'w', encoding='utf-8', newline='') as outfile:
    tsv_writer = csv.writer(outfile, delimiter='\t')
    for row in TRAIN:
        tsv_writer.writerow(row)

print(f"Combined TRAIN set written to 'Combined_Train.tsv' with {len(TRAIN)} unique rows.")
