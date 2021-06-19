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


from django.core.files.base import ContentFile

from typing import TextIO
import requests
import random	 
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
#clustering packages


nation=""
countries_list = ['United Arab Emirates','United States', 'Argentina', 'Austria', 'Australia', 'Belgium', 'Bulgaria', 'Brazil', 'Canada', 'Switzerland', 'China', 'Colombia', 'Cuba', 'The Czech Republic', 'Germany', 'Egypt', 'France', 'United Kingdom', 'Britain', 'Greece', 'Hong Kong', 'Hungary', 'Indonesia', 'Ireland', 'Israel', 'India', 'Italy', 'Japan', 'Korea', 'Lithuania', 'Latvia', 'Morocco', 'Mexico', 'Malaysia', 'Nigeria', 'Netherlands', 'Norway', 'New Zealand', 'Philippines', 'Poland', 'Portugal', 'Romania', 'Serbia', 'Russia', 'Saudi Arabia', 'Sweden', 'Slovenia', 'Slovakia', 'Thailand', 'Turkey', 'Taiwan', 'Ukraine', 'United States of America', 'Venezuela', 'South Africa']
countries = {"United Arab Emirates":"https://newsapi.org/v2/top-headlines?country=ae&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Argentina":"https://newsapi.org/v2/top-headlines?country=ar&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Austria":"https://newsapi.org/v2/top-headlines?country=at&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Australia":"https://newsapi.org/v2/top-headlines?country=au&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Belgium":"https://newsapi.org/v2/top-headlines?country=india&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3","Bulgaria":"https://newsapi.org/v2/top-headlines?country=bg&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3","Brazil":"https://newsapi.org/v2/top-headlines?country=br&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Canada":"https://newsapi.org/v2/top-headlines?country=ca&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Switzerland":"https://newsapi.org/v2/top-headlines?country=ch&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "China":"https://newsapi.org/v2/top-headlines?country=cn&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Colombia":"https://newsapi.org/v2/top-headlines?country=co&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Cuba":"https://newsapi.org/v2/top-headlines?country=cu&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "The Czech Republic":"https://newsapi.org/v2/top-headlines?country=cz&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Germany":"https://newsapi.org/v2/top-headlines?country=de&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Egypt":"https://newsapi.org/v2/top-headlines?country=eg&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "France":"https://newsapi.org/v2/top-headlines?country=fr&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "United Kingdom":"https://newsapi.org/v2/top-headlines?country=gb&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3",'United States':"https://newsapi.org/v2/top-headlines?country=gb&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Britain":"https://newsapi.org/v2/top-headlines?country=gb&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Greece":"https://newsapi.org/v2/top-headlines?country=gr&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Hong Kong":"https://newsapi.org/v2/top-headlines?country=hk&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Hungary":"https://newsapi.org/v2/top-headlines?country=hu&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Indonesia":"https://newsapi.org/v2/top-headlines?country=id&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Ireland":"https://newsapi.org/v2/top-headlines?country=ie&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Israel":"https://newsapi.org/v2/top-headlines?country=il&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "India":"https://newsapi.org/v2/top-headlines?country=in&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Italy":"https://newsapi.org/v2/top-headlines?country=it&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Japan":"https://newsapi.org/v2/top-headlines?country=jp&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Korea":"https://newsapi.org/v2/top-headlines?country=kr&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Lithuania":"https://newsapi.org/v2/top-headlines?country=lt&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Latvia":"https://newsapi.org/v2/top-headlines?country=lv&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Morocco":"https://newsapi.org/v2/top-headlines?country=ma&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Mexico":"https://newsapi.org/v2/top-headlines?country=mx&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Malaysia":"https://newsapi.org/v2/top-headlines?country=my&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Nigeria":"https://newsapi.org/v2/top-headlines?country=ng&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Netherlands":"https://newsapi.org/v2/top-headlines?country=nl&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Norway":"https://newsapi.org/v2/top-headlines?country=no&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "New Zealand":"https://newsapi.org/v2/top-headlines?country=nz&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Philippines":"https://newsapi.org/v2/top-headlines?country=ph&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Poland":"https://newsapi.org/v2/top-headlines?country=pl&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Portugal":"https://newsapi.org/v2/top-headlines?country=pt&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Romania":"https://newsapi.org/v2/top-headlines?country=ro&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Serbia":"https://newsapi.org/v2/top-headlines?country=rs&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Russia":"https://newsapi.org/v2/top-headlines?country=ru&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Saudi Arabia":"https://newsapi.org/v2/top-headlines?country=sa&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Sweden":"https://newsapi.org/v2/top-headlines?country=se&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Slovenia":"https://newsapi.org/v2/top-headlines?country=si&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Slovakia":"https://newsapi.org/v2/top-headlines?country=sk&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Thailand":"https://newsapi.org/v2/top-headlines?country=th&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Turkey":"https://newsapi.org/v2/top-headlines?country=tr&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Taiwan":"https://newsapi.org/v2/top-headlines?country=tw&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Ukraine":"https://newsapi.org/v2/top-headlines?country=ua&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "United States of America":"https://newsapi.org/v2/top-headlines?country=us&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "Venezuela":"https://newsapi.org/v2/top-headlines?country=ve&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3", "South Africa":"https://newsapi.org/v2/top-headlines?country=za&pageSize=100&apiKey=2bbf74015a094a04a118c88b58776ad3"}

# Create your views here.

def report(request):
	if request.method == 'POST':
		if request.POST.get('country') and request.POST.get('place') and request.POST.get('lat') and request.POST.get('lon'):
			global nation
			nation=request.POST.get('country')
			if nation not in countries_list:
				nation=random.choice(countries_list)
			main_url = countries[nation]
			news_result = requests.get(main_url).json() 
			article = news_result["articles"]
			newses = []
			for ar in article:
				k=ar["description"]
				if k==None:
					continue
				newses.append(k)
			sequence=random.choice(newses)
			tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
			model = GPT2LMHeadModel.from_pretrained('gpt2')
			sequence=str(sequence)
			inputs = tokenizer.encode(sequence, return_tensors='pt')
			outputs = model.generate(inputs, max_length=1000, do_sample=True)
			text = tokenizer.decode(outputs[0], skip_special_tokens=True)
			return render(request, 'news/report.html',{'file_content': text})
	else:
		return render(request, 'news/report.html')