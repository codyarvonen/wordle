import csv
import random

def select_random_word(csv_file):
    words = []
    probabilities = []

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row if it exists

        for row in reader:
            word = row[0]
            probability = float(row[1])
            if probability > 1e-5:
                probability = 1e-5
            words.append(word)
            probabilities.append(probability)

    # Normalize probabilities to ensure they add up to one
    total_probability = sum(probabilities)
    probabilities = [p / total_probability for p in probabilities]

    # Select a random word based on the probabilities
    selected_word = random.choices(words, probabilities)[0]
    return selected_word


def sum_probabilities(csv_file):
    prob_sum = 0
    count = 0
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row if it exists

        for row in reader:
            # Assuming the second column is always a number
            number = float(row[1])
            prob_sum += number
            count += 1

    return prob_sum, count

if __name__ == "__main__":
    csv_file_path = 'words.csv'
    total_prob_sum, total_count = sum_probabilities(csv_file_path)
    print(f'Total sum of probabilites: {total_prob_sum}')
    print(f'Number of columns: {total_count}')

    random_word = select_random_word(csv_file_path)
    print(f'Selected random word: {random_word}')