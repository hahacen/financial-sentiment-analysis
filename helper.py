import meta_parameters
import csv
import pandas as pd

# TODO: not sure if csv needed to be read the path
# file in is either txt or csv
def parsing(file_in, mode='pred'):
    # Use a default-dict to group values by key
    from collections import defaultdict
    grouped_data = defaultdict(list)

    # in prediction mode
    if mode == 'pred':
        txt_file = file_in
        csv_file = 'preprocessesd_pred.csv'
        lines = txt_file.split('\n')
        for line in lines:
            if '：' in line:
                parts = line.split('：')
                if len(parts[0]) >= 4:
                    key = parts[0][:4]
                else:
                    key = parts[0]
                value = parts[1]
                # Append the value to the list of values for the key
                grouped_data[key].append(value)
    # in train mode
    else:
        cvs_in = pd.read_csv(file_in)
        csv_file = 'preprocessesd_train.csv'
        titles = cvs_in['TITLE'].tolist()
        stocks = cvs_in['STOCK_NAME'].tolist()
        # TODO: not sure if zip works here
        for line, stock in zip(titles, stocks):
            key = stock
            if '：' in line:
                parts = line.split('：')
                value = parts[1]
            else:
                value = line
                # Append the value to the list of values for the key
            grouped_data[key].append(value)

    # remove duplicate
    data_processed = [(key, ' '.join(values)) for key, values in grouped_data.items()]
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['stock', 'description'])
        for entry in data_processed:
            writer.writerow(entry)

    return csv_file


def _score_calculator(sia, text: str) -> dict[str, float]:
    return sia.polarity_scores(text)


def _score_processor(dict: dict[str, float]) -> tuple[float, ...]:
    return tuple(dict.values())


def _custom_lexicon_fn():
    custom_lexicon = meta_parameters._custom_lexicon
    for word in list(custom_lexicon.keys()):
        sentiment_score = custom_lexicon[word]
        synonyms = []
        for synset in meta_parameters.wordnet.synsets(word):
            for lemma in synset.lemmas():
                synonyms.append(lemma.name())
        for synonym in synonyms:
            custom_lexicon[synonym] = sentiment_score
    return custom_lexicon


def _read_file(file_path):
    with open(file_path, 'r') as file:
        contents = file.read()
    return contents


def print_csv(file_path):
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            print(row)
