import pandas as pd 
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import preprocessor as p
from dateutil.parser import parse
import numpy as np
from nltk.tokenize import TweetTokenizer
import string
import re

tknzr = TweetTokenizer()
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.MENTION, p.OPT.RESERVED)

def cleanseRow(index, row):
	try:
		s0 = row['text']
		s0 = text.decode('utf-8').encode('ascii', 'ignore')
	except: s0 = ""
	s0 = s0.replace("#", "")
	s0 = s0.replace(".", "")
	s0 = s0.replace("'", "")
	s0 = ''.join([i for i in s0 if not i.isdigit()])
	x = p.clean(s0.lower())
	x = re.sub(r'(.)\1+', r'\1\1', x)  
	x = tknzr.tokenize(x)

	x = [i for i in x if (i not in string.punctuation)]
	x = [i for i in x if ('sandy' not in i)]
	x = [ps.stem(w) for w in x if w not in stop_words]
	x = " ".join(x)
	print(x)
	return x

data = pd.read_csv('data/text_data_copy.csv')
data['tokens'] = ""

for index, row in data.iterrows():
	print(index)
	text = row['text']
	text = text.strip("\t")
	text = text.strip("\n")
	text = text.strip("\r")
	text = text.strip("\r\n")
	data.loc[index, 'text'] = text
	data.loc[index, 'tokens'] = cleanseRow(index, row)

data = data.replace(np.nan, "", regex=True)
data = data[data.tokens != ""]
data = data[data.text != ""]
cols = ['text', 'tokens']
data.to_csv("data/cleansed_data.csv", index=False, columns=cols, header=cols)
print("saved data!!!!")
