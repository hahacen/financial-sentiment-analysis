import meta_parameters
import csv
import pandas as pd
from googletrans import Translator
# Use a default-dict to group values by key
from collections import defaultdict


# TODO: not sure if csv needed to be read the path

# file in is either txt or csv
def parsing(file_in, mode='train'):
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
        cvs_filtered = cvs_in[cvs_in['ORGAN_RATING_CONTENT'].isin(['强裂推荐', '强推', '谨慎推荐', '中性',
                                                                   'sell', '卖出', 'SELL', 'Neutral', '减持',
                                                                   'Reduce'])]
        csv_file = 'preprocessesd_train.csv'
        titles = cvs_filtered['TITLE'].tolist()  # description
        stocks = cvs_filtered['STOCK_NAME'].tolist()
        levels = cvs_filtered['ORGAN_RATING_CONTENT'].tolist()  # 评级
        # TODO: not sure if zip works here
        for title, stock, level in zip(titles, stocks,levels):
            key = stock
            if '：' in title:
                parts = title.split('：')
                value = parts[1]
            else:
                value = title
                # Append the value to the list of values for the key
            grouped_data[key].append((value, level))

    # remove duplicate
    data_processed = [(key, ' '.join([value[0] for value in values]), [value[1] for value in values][0])
                      for key, values in grouped_data.items()]
    print(data_processed)
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['stock', 'description', 'label'])
        for entry in data_processed:
            writer.writerow(entry)
    # return a path
    return csv_file


def _score_calculator(sia, text: str, is_english=False) -> dict[str, float]:
    text0 = text
    # if it's english, then no need to translate it
    # if is_english is False:
    #     translator = Translator()
    #     text0 = translator.translate(text, timeout=20)
    #     print(text0.text) #for debug use
    #     return sia.polarity_scores(text0.text)
    # else:
    return sia.polarity_scores(text0)


def _score_processor(dict: dict[str, float]) -> tuple[float, ...]:
    return tuple(dict.values())


def score_tuple(text: str, sia=meta_parameters.sia):
    return _score_processor(_score_calculator(sia, text))


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


# update the sentiment lexicon
meta_parameters.sia.lexicon.update(_custom_lexicon_fn())
