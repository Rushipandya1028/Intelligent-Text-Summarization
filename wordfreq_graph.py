# wordfreq_graph.py

from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords

def generate_word_frequency_graph(text):
    # Tokenize the text into words
    words = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.casefold() not in stop_words]
    
    # Calculate word frequencies
    freq_dist = FreqDist(words)
    
    # Get the top 20 words and their frequencies
    top_words = [word for word, _ in freq_dist.most_common(20)]
    frequencies = [freq_dist[word] for word in top_words]

    return top_words, frequencies
