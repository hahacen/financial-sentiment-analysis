import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import jax

from nltk.corpus import wordnet
# training set for sentiment evaluation
nltk.download('vader_lexicon')
nltk.download('wordnet')

epsilon = 0.01
learning_rate = 0.01
_custom_lexicon = {
    'improvement': 2.0,
    'highly recommend': 1.5,
    'not good': -1.0,
    'exceed expectations': 3,
    'cost is high': -10,
    'high growth': 5,
    'big increase': 1,
    'strong improvement': 10,
    'rapid growth': 1,
    'strong': 15,
    'accelerated release period of performance': 15,
    'completed': 10,
    'leader': 10
}
rng = jax.random.PRNGKey(0)
# choose the sentiment analyzer
# it may be used by other framework
sia = SentimentIntensityAnalyzer()
