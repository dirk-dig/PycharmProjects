#import re
#mySent='This book is the best book on Python or M.L. I have ever laid eyes upon.'
#emailText = open('email/ham/6.txt').read()
#regEx = re.compile('\\W*')
#listOfTokens = regEx.split(emailText)
#[tok.lower() for tok in listOfTokens if len(tok) > 0]
#print listOfTokens

import bayes
#reload(bayes)
#print bayes.spamTest()

import feedparser
ny=feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
