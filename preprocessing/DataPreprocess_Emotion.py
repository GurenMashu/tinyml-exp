import os
import re
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split

SAVE_DIR = "data/preprocessed_emotion_data"      # https://www.kaggle.com/datasets/nelgiriyewithana/emotions
os.makedirs(SAVE_DIR, exist_ok=True)
 
nltk.download("punkt")
nltk.download('stopwords')

df = pd.read_csv("datasets/emotion_text.csv")

df.rename(columns={"text": "Text", "label": "Label"}, inplace=True)
df.drop("Unnamed: 0", axis=1, inplace=True)

df["Text"] = df["Text"].str.replace(r"https\S+", '', regex=True)    # removing urls
df["Text"] = df["Text"].str.replace(r"[^\w\s]", '', regex=True)     # removing special chars and symbols  

df["Text"] = df["Text"].str.replace(r"\s+", '', regex=True)     # removing whitespaces
df["Text"] = df["Text"].str.replace(r"\d", '', regex=True)      # removing numeric values
df["Text"] = df["Text"].str.lower()

#removing stopwords
stop = stopwords.words("english")
df["Text"] = df["Text"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

df['Text'] = df['Text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))    # removing non-alphanumeric chars

train_df, test_df = train_test_split(df, test_size=0.2, random_state=77)

#saving 
for df, name in zip([train_df, test_df], ["train","test"]):
    df.to_csv(os.path.join(SAVE_DIR, f"{name}.csv"), index=False)