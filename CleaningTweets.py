import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

def tweet_to_words(raw_tweets):
    # Function to convert a raw tweet to a string of words
    # The input is a single string (a raw tweet), and
    # the output is a single string (a preprocessed tweet)
    #
    # 1. Remove HTML
    tweet_text = BeautifulSoup(raw_tweets,"lxml").get_text()
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", tweet_text)
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    # 4. In Python, searching a set is much faster than searching a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    # 6. Join the words back into one string separated by space and return the result.
    return( " ".join( meaningful_words ))