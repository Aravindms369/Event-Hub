from django.shortcuts import render,redirect
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse
from django.contrib.auth.forms import  AuthenticationForm
from django.contrib.auth import login, logout, authenticate
from django.contrib import messages
from django.db.models import Count
from .forms import NewUserForm
from .models import location
from django.db.models import Q
import tweepy
from tweepy import OAuthHandler

import string
import csv
import pandas as pd
import preprocessor as p
import sys
import datetime
import random
from html.parser import HTMLParser
#separate the words
import re
#Standardizing and Spell Check
import itertools
from autocorrect import Speller
from textblob import TextBlob

import requests	

import pickle
import tensorflow as tf


from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM, Conv1D, MaxPool1D, Bidirectional
from tensorflow.keras.models import Model
import numpy as np

from django.core.files.base import ContentFile
import io

#clustering packages
import nltk
from nltk.corpus import stopwords
import string
from sklearn.metrics import jaccard_score

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from wordcloud import WordCloud, STOPWORDS
nltk.download('stopwords')
nltk.download('wordnet')


nation=""
countries_list = ['United Arab Emirates','United States', 'Argentina', 'Austria', 'Australia', 'Belgium', 'Bulgaria', 'Brazil', 'Canada', 'Switzerland', 'China', 'Colombia', 'Cuba', 'The Czech Republic', 'Germany', 'Egypt', 'France', 'United Kingdom', 'Britain', 'Greece', 'Hong Kong', 'Hungary', 'Indonesia', 'Ireland', 'Israel', 'India', 'Italy', 'Japan', 'Korea', 'Lithuania', 'Latvia', 'Morocco', 'Mexico', 'Malaysia', 'Nigeria', 'Netherlands', 'Norway', 'New Zealand', 'Philippines', 'Poland', 'Portugal', 'Romania', 'Serbia', 'Russia', 'Saudi Arabia', 'Sweden', 'Slovenia', 'Slovakia', 'Thailand', 'Turkey', 'Taiwan', 'Ukraine', 'United States of America', 'Venezuela', 'South Africa']
countries = {"United Arab Emirates":"https://newsapi.org/v2/top-headlines?country=ae&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Argentina":"https://newsapi.org/v2/top-headlines?country=ar&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Austria":"https://newsapi.org/v2/top-headlines?country=at&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Australia":"https://newsapi.org/v2/top-headlines?country=au&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Belgium":"https://newsapi.org/v2/top-headlines?country=india&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3","Bulgaria":"https://newsapi.org/v2/top-headlines?country=bg&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3","Brazil":"https://newsapi.org/v2/top-headlines?country=br&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Canada":"https://newsapi.org/v2/top-headlines?country=ca&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Switzerland":"https://newsapi.org/v2/top-headlines?country=ch&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "China":"https://newsapi.org/v2/top-headlines?country=cn&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Colombia":"https://newsapi.org/v2/top-headlines?country=co&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Cuba":"https://newsapi.org/v2/top-headlines?country=cu&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "The Czech Republic":"https://newsapi.org/v2/top-headlines?country=cz&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Germany":"https://newsapi.org/v2/top-headlines?country=de&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Egypt":"https://newsapi.org/v2/top-headlines?country=eg&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "France":"https://newsapi.org/v2/top-headlines?country=fr&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "United Kingdom":"https://newsapi.org/v2/top-headlines?country=gb&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3",'United States':"https://newsapi.org/v2/top-headlines?country=gb&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Britain":"https://newsapi.org/v2/top-headlines?country=gb&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Greece":"https://newsapi.org/v2/top-headlines?country=gr&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Hong Kong":"https://newsapi.org/v2/top-headlines?country=hk&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Hungary":"https://newsapi.org/v2/top-headlines?country=hu&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Indonesia":"https://newsapi.org/v2/top-headlines?country=id&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Ireland":"https://newsapi.org/v2/top-headlines?country=ie&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Israel":"https://newsapi.org/v2/top-headlines?country=il&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "India":"https://newsapi.org/v2/top-headlines?country=in&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Italy":"https://newsapi.org/v2/top-headlines?country=it&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Japan":"https://newsapi.org/v2/top-headlines?country=jp&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Korea":"https://newsapi.org/v2/top-headlines?country=kr&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Lithuania":"https://newsapi.org/v2/top-headlines?country=lt&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Latvia":"https://newsapi.org/v2/top-headlines?country=lv&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Morocco":"https://newsapi.org/v2/top-headlines?country=ma&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Mexico":"https://newsapi.org/v2/top-headlines?country=mx&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Malaysia":"https://newsapi.org/v2/top-headlines?country=my&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Nigeria":"https://newsapi.org/v2/top-headlines?country=ng&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Netherlands":"https://newsapi.org/v2/top-headlines?country=nl&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Norway":"https://newsapi.org/v2/top-headlines?country=no&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "New Zealand":"https://newsapi.org/v2/top-headlines?country=nz&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Philippines":"https://newsapi.org/v2/top-headlines?country=ph&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Poland":"https://newsapi.org/v2/top-headlines?country=pl&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Portugal":"https://newsapi.org/v2/top-headlines?country=pt&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Romania":"https://newsapi.org/v2/top-headlines?country=ro&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Serbia":"https://newsapi.org/v2/top-headlines?country=rs&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Russia":"https://newsapi.org/v2/top-headlines?country=ru&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Saudi Arabia":"https://newsapi.org/v2/top-headlines?country=sa&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Sweden":"https://newsapi.org/v2/top-headlines?country=se&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Slovenia":"https://newsapi.org/v2/top-headlines?country=si&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Slovakia":"https://newsapi.org/v2/top-headlines?country=sk&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Thailand":"https://newsapi.org/v2/top-headlines?country=th&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Turkey":"https://newsapi.org/v2/top-headlines?country=tr&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Taiwan":"https://newsapi.org/v2/top-headlines?country=tw&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Ukraine":"https://newsapi.org/v2/top-headlines?country=ua&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "United States of America":"https://newsapi.org/v2/top-headlines?country=us&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Venezuela":"https://newsapi.org/v2/top-headlines?country=ve&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "South Africa":"https://newsapi.org/v2/top-headlines?country=za&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3"}
documents = ["covid corona cases increasing hospital vaccines covid19 cases in corona virus hospitals diesease spread",
             "covid19 cases in corona virus spreading and vaccine hospital oxygen supply virus spread",
             "government election political party election poll results coming voting in booth vote now political parties voting",
             "government election vote and poll results and parties are discussing about elected candidates vote election parties congress",
             "fifa goal scoring fifa record and messi and ronaldo greatest footballers and football fifa matches draw now and football matches goal scored euro worldcup copa",
             "fifa world cup score fifa more goals and fifa ballondor ranking fifa awards and fifa matches football soccer news football euro worldcup copa",
             "i love you cricket cricketers wickets wicket festivals drink bitcoin crypto stock market alcohol cricket wicket  shameless",
             "crypto economic financial cricketers wickets wicket stock market crypto bitcoin finance love and alcohol shameless cricket wicket"]
HashValue = ["covid","cricket","football","crypto","election"]
# Create your views here.
def home(request):
    return render(request,'news/home.html')
def map(request):
	if request.method == 'POST':
		if request.POST.get('country') and request.POST.get('place') and request.POST.get('lat') and request.POST.get('lon'):
			saverecord=location()
			saverecord.country=request.POST.get('country')
			saverecord.place=request.POST.get('place')
			saverecord.lat=request.POST.get('lat')
			saverecord.lon=request.POST.get('lon')

			auth = OAuthHandler("<enter your tokens of twitter>")
			auth.set_access_token( "<enter your tokens of twitter>")

			api = tweepy.API(auth,wait_on_rate_limit=True,wait_on_rate_limit_notify=True)
			#tweets = api.home_timeline()
			StartDate = "2021-01-01"
			latitude= float(saverecord.lat)
			longitude= float(saverecord.lon)
			max_range= 100
			lis=[]
			tweetss=[]

			with open('news/tokenizer.pickle', 'rb') as handle:
				tokenizer = pickle.load(handle)
			model = tf.keras.models.load_model('news/my_model.h5')
			model.load_weights('news/model.h5')

			for hash in HashValue:
				for tweet in tweepy.Cursor(api.search,q=hash,count=20,lang="en",since=StartDate, geocode= "%f,%f,%dkm" % (latitude, longitude, max_range), tweet_mode='extended').items(10):
					tweet=tweet.full_text
					tweet=re.sub("@[_A-Za-z0-9]+","", tweet)
					tweet=re.sub(r"http\S+", "", tweet)
					tweet=HTMLParser().unescape(tweet)

					Apos_dict={"'s":" is","n't":" not","'m":" am","'ll":" will","'d":" would","'ve":" have","'re":" are"}
					for key,value in Apos_dict.items():
						if key in tweet:
							tweet=tweet.replace(key,value)
							tweet = re.sub(r'#', '', tweet)

					tweet = re.sub(r'^RT[\s]+', '', tweet)
					tweet = " ".join([s for s in re.split("([A-Z][a-z]+[^A-Z]*)",tweet) if s])
					tweet=tweet.lower()
					file=open("news/slang.txt","r")
					slang=file.read()
					slang=slang.split('\n')
					tweet_tokens=tweet.split()
					slang_word=[]
					meaning=[]
					for line in slang:
						temp=line.split("=")
						slang_word.append(temp[0])
						meaning.append(temp[-1])

					for i,word in enumerate(tweet_tokens):
						if word in slang_word:
							idx=slang_word.index(word)
							tweet_tokens[i]=meaning[idx]
							tweet=" ".join(tweet_tokens)
					def remove_emoji(string):
						emoji_pattern = re.compile("["
									u"\U0001F600-\U0001F64F"  # emoticons
									u"\U0001F300-\U0001F5FF"  # symbols & pictographs
									u"\U0001F680-\U0001F6FF"  # transport & map symbols
									u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
									u"\U00002500-\U00002BEF"  # chinese char
									u"\U00002702-\U000027B0"
									u"\U00002702-\U000027B0"
									u"\U000024C2-\U0001F251"
									u"\U0001f926-\U0001f937"
									u"\U00010000-\U0010ffff"
									u"\u2640-\u2642"
									u"\u2600-\u2B55"
									u"\u200d"
									u"\u23cf"
									u"\u23e9"
									u"\u231a"
									u"\ufe0f"  # dingbats
									u"\u3030"
									"]+", flags=re.UNICODE)
						return emoji_pattern.sub(r'', string)

					tweet=remove_emoji(tweet)
					tweetss.append(tweet)
			
			tweets=[]
			for i in tweetss:
				if i==None:
					continue
				tweets.append(i)

			
			main_url = countries[saverecord.country]
			news_result = requests.get(main_url).json() 
			article = news_result["articles"]
			newses = []
			for ar in article:
				k=ar["description"]
				if k==None:
					continue
				newses.append(k)
			
			model_data=[]
			model_data=tweets+newses
			vectorizer = TfidfVectorizer(stop_words='english')
			X = vectorizer.fit_transform(documents)
			true_k = 4
			cluster_model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
			cluster_model.fit(X)

			order_centroids = cluster_model.cluster_centers_.argsort()[:, ::-1]
			terms = vectorizer.get_feature_names()
			for i in range(true_k):
				for ind in order_centroids[i, :10]:
					if terms[ind]=="crypto":
						clu=i

			final_lis=[]
			for i in model_data:
				a=[]
				a.append(i)
				Y = vectorizer.transform(a)
				prediction = cluster_model.predict(Y)
				if prediction == clu:
					continue
				final_lis.append(i)

			for k in final_lis:
				if k not in lis:
					j=k
					uslist=[]
					uslist.append(k)
					seq = tokenizer.texts_to_sequences(uslist)
					padded = pad_sequences(seq, maxlen= 10)
					pred = model.predict(padded)
					labels = ['diesease', 'election','football']
					eve = labels[np.argmax(pred)]

					txt=location(country=saverecord.country, place=saverecord.place, lat=saverecord.lat, lon=saverecord.lon, text= j, event=eve)
					lis.append(txt)
			location.objects.bulk_create(lis)

			return HttpResponseRedirect(reverse("news:map"))
	else:
		return render(request,'news/map.html')

def signupform(request):
	if request.method == "POST":
		form = NewUserForm(request.POST)
		if form.is_valid():
			user = form.save()
			username = form.cleaned_data.get('username')
			messages.success(request, f"New account created: {username}")
			return HttpResponseRedirect(reverse("news:loginform"))
		else:
			for msg in form.error_messages:
				messages.error(request,f"{msg}:{form.error_messages[msg]}")
	form = NewUserForm
	return render(request,"news/signupform.html", context={"form":form})
def logout_request(request):
	logout(request)
	messages.info(request, "Logged out succesfully!")
	return HttpResponseRedirect(reverse("news:loginform"))
def loginform(request):
	if request.method == 'POST':
		form = AuthenticationForm(request=request, data=request.POST)
		if form.is_valid():
			username = form.cleaned_data.get('username')
			password = form.cleaned_data.get('password')
			user = authenticate(username=username, password=password)
			if user is not None:
				login(request, user)
				messages.info(request, f"You are now logged in as {username}")
				return HttpResponseRedirect(reverse("news:userhome"))
			else:
				messages.error(request, "Invalid username or password.")
		else:
			messages.error(request, "Invalid username or password.")

	form = AuthenticationForm()
	return render(request,"news/loginform.html", context={"form":form})
def userhome(request):
	auth = OAuthHandler( "<twitter api>")
	auth.set_access_token( "<twitter api>")

	api = tweepy.API(auth)
	public_tweets = api.home_timeline()
	return render(request, 'news/userhome.html', {'public_tweets': public_tweets})

def usermap(request):
    return render(request,'news/usermap.html')


def analysis(request):
	global nation
	ob1=location.objects.filter(event="diesease").filter(country=nation)
	count1=location.objects.filter(event="diesease").filter(country=nation).count()
	ob2=location.objects.filter(event="election").filter(country=nation)
	count2=location.objects.filter(event="election").filter(country=nation).count()
	ob3=location.objects.filter(event="football").filter(country=nation)
	count3=location.objects.filter(event="football").filter(country=nation).count()
	return render(request, 'news/analysis.html', {'tw1':ob1,'tw2':ob2,'tw3':ob3,'c1':count1,'c2':count2,'c3':count3})

def report(request):
	if request.method == 'POST':
		if request.POST.get('country') and request.POST.get('place') and request.POST.get('lat') and request.POST.get('lon'):
			global nation
			nation=request.POST.get('country')
			if nation not in countries_list:
				nation=random.choice(countries_list)
			obj=location.objects.filter(country=nation)
			main_url = countries[nation]
			news_result = requests.get(main_url).json() 
			article = news_result["articles"] 
			with open("news/copy.txt", "w", encoding="utf16") as file:
				for ar in article:
					if ar["description"]==None:
						continue
					else:
						def remove_emoji(string):
							emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                u"\U00002500-\U00002BEF"  # chinese char
                                u"\U00002702-\U000027B0"
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                u"\U0001f926-\U0001f937"
                                u"\U00010000-\U0010ffff"
                                u"\u2640-\u2642"
                                u"\u2600-\u2B55"
                                u"\u200d"
                                u"\u23cf"
                                u"\u23e9"
                                u"\u231a"
                                u"\ufe0f"  # dingbats
                                u"\u3030"
								"]+", flags=re.UNICODE)
							return emoji_pattern.sub(r'', string)
						text_format = remove_emoji(ar["description"])
						file.write(text_format)
			f = open('news/copy.txt', 'r')
			file_content = f.read()
			f.close()
			return render(request, 'news/report.html',{'file_content': file_content})
	else:
		return render(request, 'news/report.html')
