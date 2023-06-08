import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import jax

# TODO: add weight on the words that appears more time, +label*(some value), learn the weights instead of do it myself
# TODO: try rnn to train the lexicon myself in Chinese, (labeled needed? Can we have unsupervised learning?)
from nltk.corpus import wordnet
# training set for sentiment evaluation
nltk.download('vader_lexicon')
nltk.download('wordnet')

epsilon = 0.01
learning_rate = 5e-5
_custom_lexicon = {
    'highly recommend': 10.5,
    'not good': 10.0,
    'exceed expectations': 10,
    'cost is high': 10,
    'high growth': 5,
    'big increase': 10,
    'strong improvement': 15,
    'rapid growth': 25,
    'strong': 20,
    'accelerated release period of performance': 15,
    'completed': 10,
    'leader': 10,
    'great pressure': -25,
    'slow down': -25,
    'suffer significantly less': 20,
    'too optimistic': -10,
    'booming': 5,
    'quickly increase': 10,
    # # 'slightly': -10,
    # # 'small': -10,
    # # 'expected to continue': -10,
    # # 'expected to grow': -10,
    'outstanding': 10,
    # 'look forward to': -10,
    'fell short of expectation': -25,
    'dragging down': -30,
    'hedge': 10,
    'Valuation has fallen': -10,
    'performance declined': -18,
    'Excellent performance': 10,
    'improved significantly': 15,
    'verified': 8,
    'transparent': 8,
    'neutral': -5,
    'fluctuates violently': -15,
    'affect': -18,
    'new high': 10,
    'improve significant': 20,
    'impact': -20,
    'recover': 8,
    'more than expected': 30,
    'exceeded expectations': 30,
    'accelerate': 15,
    'poor': -20



}
rng = jax.random.PRNGKey(0)
# choose the sentiment analyzer
# it may be used by other framework
sia = SentimentIntensityAnalyzer()
helper_dic = {
    "强裂推荐": 1,
    "强推": 1,
    "谨慎推荐": 0,
    "中性": 0,
    "sell": -1,
    "卖出": -1,
    "SELL": -1,
    "Neutral": 0,
    "减持": -1,
    "Reduce": -1
}
