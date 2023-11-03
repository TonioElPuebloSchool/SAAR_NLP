"""
Explanations concerning the choices made in the preprocessing pipeline -
are provided in the 2_preprocessing_pipeline_notebook.ipynb file.
"""

import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

stop_words = set(stopwords.words("english"))

def lower_case(text: str) -> str:
    """
    _summary_

    Args:
        text (str): text to lower case

    Returns:
        str: text lower cased
    """
    return text.lower()
def remove_stop_words(text: str) -> str:
    """_summary_

    Args:
        text (str): text to remove stop words from

    Returns:
        str: text with stop words removed
    """
    Text=[i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)
def remove_numbers(text: str) -> str:
    """_summary_

    Args:
        text (_type_): text to remove numbers from

    Returns:
        _type_: text with numbers removed
    """
    text=''.join([i for i in text if not i.isdigit()])
    return text
def remove_punctuations(text: str) -> str:
    """_summary_
    
    Args:
        text (_type_): text to remove punctuations from
        
    Returns:
        _type_: text with punctuations removed
    """
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛',"", )
    text = re.sub('\s+', ' ', text)
    text =  " ".join(text.split())
    return text.strip()
def remove_urls(text: str) -> str:
    """_summary_
    
    Args:
        text (_type_): text to remove urls from
        
    Returns:
        _type_: text with urls removed
    """
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)
def lemmatization(text: str) -> str:
    """_summary_

    Args:
        text (str): text to lemmatize

    Returns:
        str: text with lemmatized words
    """
    lemmatizer= WordNetLemmatizer()
    text = text.split()
    text=[lemmatizer.lemmatize(y) for y in text]
    return " " .join(text)
def preprocessing_pipeline(text: str) -> str:
    """
    Chains all the cleaning functions together using scikit-learn pipelines.
    """
    preprocessing_steps = [
        ('lower_case', FunctionTransformer(lower_case)),
        ('remove_stopwords', FunctionTransformer(remove_stop_words)),
        ('remove_numbers', FunctionTransformer(remove_numbers)),
        ('remove_punctuation', FunctionTransformer(remove_punctuations)),
        ('remove_urls', FunctionTransformer(remove_urls)),
        ('lemmatization', FunctionTransformer(lemmatization))
    ]

    # now we can create the pipeline
    preprocessing_pipeline = Pipeline(preprocessing_steps)

    # and finally, we apply the pipeline to the input text
    cleaned_text = preprocessing_pipeline.transform([text][0])

    return cleaned_text

if __name__ == "__main__":
    df = pd.read_csv("data\clean_data\dataset.csv", index_col=0)
    df["text"] = df.text.apply(lambda x: preprocessing_pipeline(x))