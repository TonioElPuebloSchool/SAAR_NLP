import pandas as pd
import re
import string
import nltk
from spellchecker import SpellChecker
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

# Load NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

# Define preprocessing functions
def remove_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def remove_http_tags(text):
    return re.sub('<.*?>+', '', text)

def spell_correction(text):
    spell = SpellChecker()
    words = text.split()
    corrected_words = [spell.correction(word) if word in spell.unknown(words) else word for word in words]
    return ' '.join(corrected_words)

def lower_case(text):
    return text.lower()

def remove_punctuation(text):
    return re.sub('[%s]' % re.escape(string.punctuation), '', text)

def remove_stopwords(text, language='english'):
    stopwords_set = set(stopwords.words(language))
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords_set]
    return ' '.join(filtered_words)

def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    wordnet_map = {'N': wordnet.NOUN, 'V': wordnet.VERB, 'R': wordnet.ADV, 'J': wordnet.ADJ}
    pos_tagged_text = nltk.pos_tag(nltk.word_tokenize(text))
    lemmatized_words = [lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text]
    return ' '.join(lemmatized_words)

# pipeline
def preprocessing_pipeline(text: str) -> str:
    """
    Chains all the cleaning functions together using scikit-learn pipelines.
    """
    # Create the preprocessing pipeline
    preprocessing_steps = [
        ('lower_case', FunctionTransformer(lower_case)),
        ('remove_urls', FunctionTransformer(remove_urls)),
        ('remove_http_tags', FunctionTransformer(remove_http_tags)),
        ('remove_punctuation', FunctionTransformer(remove_punctuation)),
        ('remove_stopwords', FunctionTransformer(lambda x: remove_stopwords(x, 'english'))),
        #('lemmatize', FunctionTransformer(lemmatize)),
        #('spell_correction', FunctionTransformer(spell_correction))
    ]

    preprocessing_pipeline = Pipeline(preprocessing_steps)

    # Apply the pipeline to the input text
    cleaned_text = preprocessing_pipeline.transform([text][0])

    return cleaned_text

# Example usage
if __name__ == "__main__":
    df = pd.read_csv("data/clean_data/Musical_instruments_reviews_clean.csv", index_col=0)
    df["cleaned_text"] = preprocessing_pipeline.transform(df['text'])
    # Save the preprocessed DataFrame to a new CSV file if needed
    df.to_csv("data/clean_data/Musical_instruments_reviews_preprocessed.csv")
