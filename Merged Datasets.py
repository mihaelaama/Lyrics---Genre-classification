import pandas as pd
import csv
import os
import nltk
import pickle
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

from langdetect import detect_langs
from langdetect.lang_detect_exception import LangDetectException

df1 = pd.read_csv('.csv')
df2 = pd.read_csv('metal_lyrics.csv')

# Renaming columns in df1 to match the preferred naming scheme
df2.rename(columns={'Artist': 'artist', 'Song': 'title', 'Lyric': 'lyrics'}, inplace=True)

df1_selected = df1[['artist', 'title', 'lyrics']].copy()
df2_selected = df2[['artist', 'title', 'lyrics']].copy()

merged_df = pd.concat([df1_selected, df2_selected], ignore_index=True)

# Remove duplicates based on 'Title' and 'Artist'
merged_df.drop_duplicates(subset=['title', 'artist'], inplace=True)

# Drop rows with missing 'Lyrics'
merged_df.dropna(subset=['lyrics'], inplace=True)

#print(df1.columns)
#print(df2.columns)

merged_df.to_csv('D:\Disertatie\merged_dataset.csv', index=False)