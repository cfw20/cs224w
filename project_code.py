# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 12:58:57 2018

@author: User
"""


import snap
import collections
import string
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process


df = pd.read_csv('major_customers.csv', index_col='conm', parse_dates=['srcdate'])
print df.shape
df.head()
df.tail()
for i in df.columns:
    print 'column: %s, type: %s' % (i, type(i)) 

#cnms_labels, cnms_uniques = pd.factorize(df['cnms'])
#len(df['cnms'])
#len(cnms_uniques)
#conm_labels, conm_uniques = pd.factorize(df.index)
#len(df.index)
#len(conm_uniques)
#ctype_labels, ctype_uniques = pd.factorize(df['ctype'])
#len(df['ctype'])
#len(ctype_uniques)

#df.loc[:,'compNode'] = conm_labels
#df.loc[:,'custNode'] = cnms_labels
#df.loc[:,'customerType'] = ctype_labels

#only customerType == COMPANY
df = df[df['ctype'] == 'COMPANY']
print df.shape
#get rid of Not Reported customer name
df = df[df['cnms'] != 'Not Reported']
print df.shape
#get rid of # Customers 
df = df[~df['cnms'].str.contains('Customers')]
print df.shape
#get rid of duplicate index and cnms pairs
df = df.drop_duplicates(subset=['tic','cnms'],keep='first')


#load NYSE company names to tickers
nyse = pd.read_csv('nysecompanylist.csv')
nasdaq = pd.read_csv('nasdaqcompanylist.csv')
amex = pd.read_csv('amexcompanylist.csv')
namesToTicker = pd.concat([nasdaq, nyse, amex])

exactList = []
for i in df['cnms'][:30]:
#    print 'original %s' %i
#    print process.extract(i, namesToTicker['Name'], limit=3, scorer=fuzz.token_set_ratio)
    a, b, c = process.extractOne(i, namesToTicker['Name'], scorer=fuzz.token_set_ratio)
    if b == 100:
        exactList.append((i,a))