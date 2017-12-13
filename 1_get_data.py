# Wang, H., E.H. Hovy, and M. Dredze. 2015. The Hurricane Sandy Twitter Corpus. Proceedings of the AAAI Workshop on the World Wide Web and Public Health Intelligence.

import pandas as pd 
import tweepy

CONSUMER_KEY = "tKqeLqaXP3OoIoy3cmhvDKYcu"
CONSUMER_SECRET = "cg0KJ3fiCsKsqyaJAtTScyA2wTm4TEr7ErLGZ2MUPgm13A6yv8"
OAUTH_TOKEN = "710647818721996800-7qdHucUPNzwiM6Jr1HMaTLKpi7jdcBW"
OAUTH_TOKEN_SECRET = "QWOm7c1DnCdxA6k6JR5ABvExCNWt81aRIQFDEu2ri5czh"

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
# api = tweepy.API(auth)
api = tweepy.API(auth, wait_on_rate_limit = True)


data = pd.read_csv('data/sandy_filter_tweets.txt', sep="\t", header=None)
data.columns = ["id", "date", "sandy"]
data = data[data.sandy]
data['geo_enabled'] = False
data['lat'] = ""
data['long'] = ""
data['text'] = ""

# get text from tweets
print("getting tweets")
for index, row in data.iterrows():

	row_id = row['id'].split(":")[-1]
	data.loc[index, 'id'] = row_id

	try:
		text = api.get_status(int(row_id))
		data.loc[index, 'text'] = text.text.encode('utf-8')
		# data.loc[index, 'screen_name'] = text.user.screen_name
		# data.loc[index, 'geo_enabled'] = text.user.geo_enabled
		# g = text.geo
		# coords = g[u'coordinates']
		# data.loc[index, 'lat'] = coords[0]
		# data.loc[index, 'long'] = coords[1]

	except: print("rip")
	

data = data[data.text != ""]
data = data[data.geo_enabled]
data = data[data.lat != ""]
data = data[data.long != ""]

cols = ['id', 'date', 'text', 'screen_name', 'lat', 'long']
cols = ['id', 'text']
data.to_csv("data/text_data_temp.csv", index=False, columns=cols, header=cols)
print("saved data!!!!")

