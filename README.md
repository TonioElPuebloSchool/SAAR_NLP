<img style="float: left; padding-right: 10px; width: 250px" src="https://upload.wikimedia.org/wikipedia/fr/b/b1/Logo_EPF.png?raw=true" > 

![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white) 

![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

## **Natural Language Processing: Sentiment Analysis**
10-11/2023 - Created by Antoine Courbi

-----
#### [*Course instructions*](https://github.com/RPegoud/nlp_courses/blob/main/nlp_courses/tp_4_5_NLP_project/instructions.md)
#### [*Github depository*](https://github.com/TonioElPuebloSchool/SAAR_NLP)
-----

This **README** is meant to explain how the project was carried out. It is divided into the following sections:

- [**Introduction**](#introduction)
- [**Dataset**](#dataset)
- [**Requirements**](#requirements)
- [**Results**](#results)
- [**Conclusion**](#conclusion)
- [**References**](#references)

# **Introduction**

**Sentiment analysis** is a common task in **Natural Language Processing**. It consists in classifying a sentence into one of several `categories`, depending on the `sentiment` it expresses.  
Since it's a hard task to do **manually**, it's a good candidate for **automation**.  Moreover, it can be used in many applications, such as chatbots, product reviews, tweets, etc.  

This project is meant to **investigate** different methods to perform sentiment analysis, using different `models` and `techniques`.  Great models already exist, such as [`BERT`](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270), so this project is not meant to create a new incredible model, but rather to explore the different **possibilities** and to compare them.

# **Dataset**

The dataset used for this project is the [**`Emotions dataset for NLP`**](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp/code) from kaggle.  
It provides three files :
- `train.txt` : 16000 sentences with their corresponding labels
- `test.txt` : 2000 sentences with their corresponding labels
- `val.txt` : 2000 sentences with their corresponding labels
  
The labels and theire distributions on the training set are the following:
- `joy`         33.5%
- `sadness`     29.2%
- `anger`       13.5%
- `fear`        12.1%
- `love`         8.1%
- `surprise`     3.6%

# **Requirements**

In order to use the **models**, you can download the ***requirements.txt*** file and install the required packages using `anaconda` by running the following command in your terminal:
```bash
$ conda create --name <env> --file <requirements.txt>
```
That will create a new **environment** with the required packages. You can then activate the environment by running the following command:
```bash
$ conda activate <env>
```

# **Results**

The following table shows the results obtained with the different models.

| **Model**                                      | **Training Time** | **Accuracy**   |
|--------------------------------------------|---------------|------------|
| Baseline Model                             | 2.1s          | 70%        |
| Baseline Model with Balancer               | 5.1s          | 84%        |
| Logistic Regression                        | 2s            | 87%        |
| Logistic Regression with Balancer          | 5.6s          | 88%        |
| Decision Tree Classifier                   | 1mn03s        | 86%        |
| Support Vector Classifier (SVC)            | 5mn45s        | 86%        |
| Random Forest Classifier                   | 7mn10s        | 87%        |
| **LR with Balancer and Hyperparameter Tuning** | **31.5s** | **90%** |
| Manual RNN                                 | 6mn14         | 81%        |


# **Conclusion**

The best model is the **`Logistic Regression`** with **Balancer** and **Hyperparameter Tuning**, obtaining an **accuracy** of **90%**.  

The **`manual RNN`** could be improvded by using a **bidirectional** RNN, and by using a **pretrained** embedding layer, because its currently using a one-hot encoding, which is not optimal.  

Also, the **dataset** could be **augmented** by using **synonyms** and **antonyms** of the words in the sentences, which would increase the **variety** of the dataset.

# **References**
Articles:
- [BERT Explained: State of the art language model for NLP](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270)
- [Sentiment Analysis: A Practitioner’s Guide to NLP](https://monkeylearn.com/sentiment-analysis/)
- [Contextualized Affect Representations for Emotion Recognition](https://aclanthology.org/D18-1404/)
- [Practical Text Classification With Python and Keras](https://realpython.com/python-keras-text-classification/)

Notebooks:
- [BERTweet](https://github.com/VinAIResearch/BERTweet)
- [Emotions detection from tweets](https://www.kaggle.com/code/takai380/emotion-detection-from-tweets-roberta-fine-8eda50/notebook)

Github:
- [RNN pour l'analyse de sentiments](https://github.com/aminaghoul/sentiment-analysis/)
- [Bentrevett sentiment analysis](https://github.com/bentrevett/pytorch-sentiment-analysis/)
- 
Other:
- [Hugging Face](https://huggingface.co/)
- [Markdown badges](https://github.com/Ileriayo/markdown-badges)

<p align="center">&mdash; ⭐️ &mdash;</p>
<p align="center"><i>This README was created during the NLP course</i></p>
<p align="center"><i>Created by Antoine Courbi</i></p>