# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 16:36:19 2018

@author: wongchi
"""

import os
import snap
import pandas as pd
import string
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

os.getcwd()

#names = ['ticker', 'marketCap'] + list(string.ascii_lowercase)[:20]
#df = pd.read_csv('bloomberg_supply_chain.csv', names = names)
names = ['ticker'] + list(string.ascii_lowercase)[:20]
df = pd.read_csv('russell3000_supply.csv', names = names)
df.shape

#check index only contains US stocks
def checkIndex(df):
    check = []
    for i in df['ticker']:
        if any(x in i for x in 
               ['UN Equity', 'UW Equity', 'UF Equity', 'UR Equity', 'UA Equity', 'UQ Equity', 'VF Equity', 'UV Equity']) is False:
            check.append(i)
    print 'number of non US stocks: %d' % len(check)
    return check

def foreignDom(ticker):
    if type(ticker) is not str:
        return False
    us = any(x in ticker for x in 
             ['UN Equity', 'UW Equity', 'UF Equity', 'UR Equity', 'UA Equity', 'UQ Equity', 'VF Equity', 'US Equity', 'UV Equity'])
    return not us
        
def usDom(ticker):
    if type(ticker) is not str:
        return False
    us = any(x in ticker for x in 
             ['UN Equity', 'UW Equity', 'UF Equity', 'UR Equity', 'UA Equity', 'UQ Equity', 'VF Equity', 'US Equity', 'UV Equity'])
    return us

#copy dataframe
ef = df.copy(deep=True)
    
#remove US Equity etc from ticker column
ef['ticker'] = ef['ticker'].str.split().str[0]

#turn all foreign stocks to -1
ef.iloc[:,1:] = ef.iloc[:,1:].applymap(lambda x: -1 if foreignDom(x) else x)

#turn all NaN to -1
ef.iloc[:,1:] = ef.iloc[:,1:].fillna(-1)

#shorten all US ticker codes
ef.iloc[:,1:] = ef.iloc[:,1:].applymap(lambda x: x.split()[0] if usDom(x) else x)

#remove market cap column and create new df
#ef = ef.drop('marketCap', axis=1)

#save dataframe
ef.to_pickle('russell3000_pickled.pkl')

#read dataframe
ef = pd.read_pickle("russell3000_pickled.pkl")

#turn -1 to np.nan
ef[ef == -1] = np.nan

#count number of edges
ef.iloc[:,1:].count().sum()

#set up ticker set and dict
ticSet = set(ef['ticker'])
ticDict = {ef['ticker'][i]:i for i in range(len(ef['ticker']))}
invTicDict = {i:ef['ticker'][i] for i in range(len(ef['ticker']))}


#construct edges set
def edgeSet(ef, ticDict):
    edges = set()
    for row in ef.itertuples():
        for col in list(string.ascii_lowercase)[:20]:
            if getattr(row, col) in ticDict:
                edges.add((ticDict[row.ticker], ticDict[getattr(row, col)]))
    return edges

#construct graph from ticDict and edges
def constructGraph(ticDict, edges):
    G1 = snap.TNGraph.New() 
    for ticker in ticDict.keys():
        G1.AddNode(ticDict[ticker])
    for edge in edges:
        G1.AddEdge(*edge)
    return G1

#make graph g, save g
edges = edgeSet(ef, ticDict)
g = constructGraph(ticDict, edges)
# Save graph to binary
FOut = snap.TFOut('bloomberg.graph')
g.Save(FOut)
FOut.Flush()
# Load graph from binary
FIn = snap.TFIn('bloomberg.graph')
G4 = snap.TNGraph.Load(FIn) 


def egonet(graph, node):
    NIdV = snap.TIntV()
    NIdV.Add(node)
    for n in graph.GetNI(node).GetOutEdges():
        NIdV.Add(n)
#    for n in graph.GetNI(node).GetInEdges():
#        NIdV.Add(n)
    subGraph = snap.GetSubGraph(graph, NIdV)
    return subGraph

def egonet2(graph, node):
    a = snap.TIntV()
    checkSet = set()
    a.Add(node)
    checkSet.add(node)
    for n in graph.GetNI(node).GetOutEdges():
        a.Add(n)
        checkSet.add(n)
    for i in a:
        for j in graph.GetNI(i).GetOutEdges():
            if j not in checkSet:
                a.Add(j)
    subGraph = snap.GetSubGraph(graph, a)
    return subGraph

def pe(G1):
    print 'no. of nodes: %d' % G1.GetNodes()
    print 'no. of edges: %d' % G1.GetEdges()
    for EI in G1.Edges(): # Edge traversal
        print '(%d, %d)' % (EI.GetSrcNId(), EI.GetDstNId()) 
    for node in G1.Nodes():
        print node.GetId()

def plotGraph(g, invTicDict, name="output.png"):
    labels = snap.TIntStrH()
    for NI in g.Nodes():
            labels[NI.GetId()] = str(invTicDict[NI.GetId()])
    snap.DrawGViz(g, snap.gvlDot, name, " ", labels)

def plotEgonet(G, node, invTicDict): 
    subGraph = egonet(G, node)
    plotGraph(G, invTicDict)
    return subGraph

#check max strongly connected and weakly connected    
MxScc = snap.GetMxScc(g)
print 'no. of nodes: %d' % MxScc.GetNodes()
print 'no. of edges: %d' % MxScc.GetEdges()
plotGraph(MxScc, invTicDict, 'MxScc.png')

MxWcc = snap.GetMxWcc(g)
print 'no. of nodes: %d' % MxWcc.GetNodes()
print 'no. of edges: %d' % MxWcc.GetEdges()

#M5 cut
cluster = [1173, 894, 288, 947, 285, 978, 599, 729, 604, 771, 432, 470, 260]
j = 0
for node in MxWcc.Nodes():
    if j in cluster:
        print j, node.GetId(), invTicDict[node.GetId()]
    j += 1
    

#measuring pageRank
    
russPrices = pd.read_csv('russellprices.csv', parse_dates=['Ticker'])
temp = russPrices.iloc[2:,:]
temp1 = temp.set_index('Ticker')
russReturns = temp1.apply(pd.to_numeric, errors='coerce').pct_change()
corrMatrix = russReturns.iloc[1:,:].corr(min_periods=50)

xy = corrMatrix.mask(np.triu(np.ones(corrMatrix.shape)).astype(bool)).stack()
xy.mean()
xy.hist(bins=20, grid=True)
node1, node2 = zip(*xy.index)
x1 = [n.split()[0] for n in node1]
x2 = [n.split()[0] for n in node2]
id1 = [ticDict[n] for n in x1]
id2 = [ticDict[n] for n in x2]
shortPath = [snap.GetShortPath(g, id1[i], id2[i]) for i in range(len(x1))]
    
xymain = pd.DataFrame({'x1':x1, 'x2':x2, 'corr':xy, 'id1':id1, 'id2':id2, 
                       'shortPath':shortPath})

#plot hist of corrs
xymain['corr'].hist(bins=20)
plt.title('Correlation of returns of node pairs')
plt.show()

#plot shortpath vs corrs
plt.scatter(xymain['shortPath'], xymain['corr'])
plt.title('Node pairs: ShortPath vs Corr of Returns')
plt.show()


#russAdjMat = get_adjacency_matrix(g)
#m = corrMatrix.multiply(russAdjMat)
#m[m==0] = np.nan
#m.count()
#add pageRank to xymain
PRankH = snap.TIntFltH()
snap.GetPageRank(g, PRankH)
pr1 = [PRankH[i] for i in id1]
pr2 = [PRankH[i] for i in id2]

xymain['pr1'] = pr1
xymain['pr2'] = pr2




y = xymain['corr']

X = xymain[['id1','id2']]
X = xymain[['pr1','pr2','shortPath']]
#X = xymain['shortPath']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = Ridge(alpha=1.0)
clf.fit(X_train, y_train) 
clf.score(X_test, y_test)

#plot shortpath vs corrs
plt.scatter(xymain['pr1'], xymain['corr'])
plt.title('Node pairs: ShortPath vs Corr of Returns')
plt.show()



pageRankCorr = []
for node in g.Nodes():
    tot = 0.0
    inEdges = 0.0
    for k in node.GetInEdges():
        if np.isnan(corrMatrix.iloc[k, node.GetId()]) != True:
            tot += corrMatrix.iloc[k, node.GetId()]
            inEdges += 1
    if inEdges > 0:
        ave = tot / inEdges
    else:
        ave = 0.0
    pageRankCorr.append((PRankH[node.GetId()], ave))

p, c = zip(*pageRankCorr)
plt.scatter(c, p)
#trend = np.polyfit(c, p, 1)
#plt.plot(c, trend[1] + trend[0] * c)
#print "a %.2f, b %.2f" % (trend[0], trend[1])

plt.xlim(-0.5, 1)
plt.ylim(0, 0.01)
plt.ylabel('pageRank')
plt.xlabel('Ave. corr')
plt.title('pageRank vs. Ave. corr of connected nodes')
plt.show()


sorted_PRankH = sorted(PRankH, key = lambda key: PRankH[key], reverse = True)

sorted_pageRankCorr = sorted(pageRankCorr, key = lambda key: key[0], reverse=True)
print 'ticker', 'PageRank', 'Ave. Corr'
k = 0
for key in sorted_PRankH: # Iterate over keys
    if k > 5:
        break
    print invTicDict[key], PRankH[key], sorted_pageRankCorr[k][1] 
    k += 1


