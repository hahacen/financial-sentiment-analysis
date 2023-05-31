import meta_parameters
import csv


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
