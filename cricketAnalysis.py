#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ##### Import all data

# In[2]:


years = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
data = []

for year in years:
    small_df = pd.read_html("https://www.espncricinfo.com/table/series/8048/season/" + str(year) + "/indian-premier-league")
    small_df = small_df[0]
    
    small_df["TEAMS"] = small_df["TEAMS"].str.replace("\d+", "")
    small_df["FOR"] = small_df["FOR"].str[:4]
    small_df["AGAINST"] = small_df["AGAINST"].str[:4]
    small_df.insert(1, "SEASON", str(year))
    
    data.append(small_df)

df = pd.concat(data)

df.replace({"Kings XI Punjab": "Punjab Kings"}, inplace=True)
df.replace({"Delhi Daredevils": "Delhi Capitals"}, inplace=True)
df[["FOR", "AGAINST"]] = df[["FOR", "AGAINST"]].apply(pd.to_numeric)

df = df.set_index(["TEAMS", "SEASON"]).sort_index()

df.drop(["Rising Pune Supergiant", "Rising Pune Supergiants", "Pune Warriors",
         "Kochi Tuskers Kerala", "Deccan Chargers", "Gujarat Lions"], inplace=True)


# In[3]:


df


# To find season table:
# 
# > df.loc[pd.IndexSlice[:, **"year"**], :].sort_values("PT", ascending=False)
# 
# To find team results:
# 
# > df.loc[**"team"**]

# In[4]:


#  Pythagorean expectation
#  Use to calculate k using MSE

def pythagExpectForK(runsFor, runsAgainst, M, k):
    
    predWinPercent = (runsFor ** k / (runsFor ** k + runsAgainst ** k))
    predWins = round(predWinPercent * M, 0)
    
    return predWinPercent, predWins


# In[5]:


exp = np.linspace(8.05, 8.08, 301)  # Found to be between 8 and 8.2 from previous iterations

temp = []
MSE = []

for i in exp:
    for j in range(len(df)):
        
        predWinPercent, _ = pythagExpectForK(df.loc[df.index[j], "FOR"], df.loc[df.index[j], "AGAINST"],
                                         df.loc[df.index[j], "M"], i)
        
        trueWinPercent = df.loc[df.index[j], "W"] / df.loc[df.index[j], "M"]
        
        temp.append((predWinPercent - trueWinPercent) ** 2)

    MSE.append(sum(temp) / len(temp))

#  To find lowest value for k

MSE = np.array(MSE)
exp = np.array(exp)
zipped = zip(exp, MSE)

for i in zipped:
    if MSE.min() in i:
        print(i[0])    
    
plt.plot(exp, MSE)  # Visualise MSE


# ##### k = 8.062

# In[6]:


#  Pythagorean expectation using calculated k

def pythagExpect(runsFor, runsAgainst, M):
    
    predWinPercent = (runsFor ** 8.062 / (runsFor ** 8.062 + runsAgainst ** 8.062))
    predWins = round(predWinPercent * M, 0)
    
    return predWinPercent, predWins


# In[7]:


#  Test

runsFor = 2241
runsAgainst = 2195
M = 14

predWinPercent, predWins = pythagExpect(runsFor, runsAgainst, M)

print("Predicted number of wins:", predWins, "\nPredicted win %:", predWinPercent)  # Pred win needed?


# In[8]:


df["WIN %"] = (df["W"] / df["M"])
df["PRED WIN %"], _ = pythagExpect(df["FOR"], df["AGAINST"], df["M"])


# In[9]:


df


# In[10]:


teams = df.index.get_level_values("TEAMS").unique()
i = 0

fig, axs = plt.subplots(3, 3, figsize=(16, 8))
axs = axs.ravel()

for team in teams:
    
    temp_df = df.loc[team, :]
    temp_df.reset_index(inplace=True)

    axs[i].scatter(temp_df["SEASON"], temp_df["WIN %"], label="True win %", c="g", s=50)
    axs[i].scatter(temp_df["SEASON"], temp_df["PRED WIN %"], label="Pred win %", c="orange", s=50)
    
    axs[i].tick_params(axis="x", labelrotation=45)
    axs[i].set_ylim([0.2, 0.8])
    #axs[i].legend()
    axs[i].title.set_text(team)
    
    i += 1

handles, labels = axs[1].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower right', prop={'size': 20})

fig.delaxes(axs[8])
fig.tight_layout()


# ##### What now..?

# In[ ]:




