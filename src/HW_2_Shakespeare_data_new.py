
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from subprocess import check_output
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('C:\\Users\\p860n111\\Desktop\\data science\\Hw2\\Shakespeare_data.csv')


# In[3]:


df.describe


# In[6]:


#to check the values for columns that have NaN value in them 
print(df.info())


# In[10]:


df.head(10)


# In[11]:


df.describe()


# In[14]:


# to see the total number of columns in the dataset.
df.columns


# In[16]:


#to find the total number of unique play
print("Number of plays are: " + str(df['Play'].nunique()))


# In[17]:


# to replace the NaN value in the Player column to be Unknown
df['Player'].replace(np.nan, 'Unknown',inplace = True)


# In[19]:


# to check that the Player column has all i.e. 111396 non null values.
df.info()


# In[20]:


# to verify that the missing values in Palyer column got replaced by Unknown
df.head()


# In[24]:


# to print the count of unique plays and their names
print("The total number of plays are: " + str(df['Play'].nunique())+" and their names are as follows: " + str(df['Play'].unique()))


# In[25]:


#creating a new data frame that will contain unique Plays value as a new column named: Name of the Play

pd.DataFrame(df['Play'].unique().tolist(), columns=['Name of the Play'])


# In[27]:


# Additional Information 1: For each Play, number of lines (PlayerLine) spoken by each PlaYer

df.groupby(['Play','Player']).count()['PlayerLine']


# In[28]:


# Now converting the above data into a frame (playWise_lines_per_player).

playWise_lines_per_player= df.groupby(['Play','Player']).count()['PlayerLine']
playWise_lines_per_player= playWise_lines_per_player.to_frame()
playWise_lines_per_player


# In[29]:


playWise_lines_per_player.describe()


# In[33]:


# Additional Information 2: To count the number of PlayerLine corresponding to each Play.

df.groupby('Play').count().sort_values(by='PlayerLine',ascending=True)['PlayerLine']


# In[44]:


#Converting the data into a dataframe (playerLinePerPlay)
playerLinePerPlay = df.groupby('Play').count().sort_values(by='PlayerLine',ascending=True)['PlayerLine']


# In[45]:


playerLinePerPlay


# In[46]:


playerLinePerPlay = playerLinePerPlay.to_frame()


# In[47]:


playerLinePerPlay


# In[48]:


# applying indexing to the above dataframe
playerLinePerPlay['Play'] = playerLinePerPlay.index.tolist()


# In[49]:


playerLinePerPlay


# In[50]:


playerLinePerPlay.index = np.arange(0,len(playerLinePerPlay))


# In[51]:


playerLinePerPlay


# In[58]:


# plotting a graph to show: PlayerLine against Name of the Play
plt.figure(figsize=(90,50))
ax= sns.barplot(x='Play',y='PlayerLine',data=playerLinePerPlay, order = playerLinePerPlay['Play'])
ax.set(xlabel='Name of the Play', ylabel='PlayerLines')
plt.show()


# In[61]:


plt.figure(figsize=(15,15))
ax= sns.barplot(x='PlayerLine',y='Play',data=playerLinePerPlay, order = playerLinePerPlay['Play'])
ax.set(xlabel='PlayerLines', ylabel='Name of the Play')
plt.show()


# In[63]:


# Additional Information 3: Number of Players corresponding to each Play

playersPerPlay = df.groupby(['Play'])['Player'].nunique().sort_values(ascending= True)


# In[64]:


playersPerPlay


# In[65]:


#changing to it to dataframe
playersPerPlay=playersPerPlay.to_frame()


# In[66]:


playersPerPlay


# In[67]:


playersPerPlay['Play'] = playersPerPlay.index.tolist()


# In[68]:


playersPerPlay


# In[69]:


# now to change the index from Play to 0 - (length-1) adn renaming the column name 

playersPerPlay.columns = ['Number of Players','Name of the Play']


# In[70]:


playersPerPlay


# In[71]:


playersPerPlay.index= np.arange(0,len(playersPerPlay))
playersPerPlay


# In[72]:


# plotting graph

plt.figure(figsize=(15,15))
ax = sns.barplot(x='Number of Players',y='Name of the Play',data=playersPerPlay)
ax.set(xlabel='Number of Players', ylabel='Name of the Play')
plt.show()


# In[77]:


plt.figure(figsize=(100,100))
ax = sns.barplot(x='Name of the Play',y='Number of Players',data=playersPerPlay)
ax.set(xlabel='Name of the Play', ylabel='Number of Players')
plt.show()


# In[82]:


# to calculate the total words in each PlayerLine cell entry
df['new_column'] = df.PlayerLine.apply(lambda x: len(str(x).split(' ')))


# In[83]:


df


# In[89]:


df.groupby(['Player'])['new_column']


# In[90]:


df


# In[93]:


df


# In[94]:


g.max


# In[96]:


g.sum()


# In[98]:


g.new_column


# In[100]:



df.rename(columns={'new_column': 'NoOfWordsInPlayerLine'}, inplace=True)


# In[101]:


df


# In[102]:


h=df.groupby('Player')


# In[103]:


for Player, data in h:
    print("Player:",Player)
    print("\n")
    print("Player:",data)


# In[104]:


h


# In[105]:


g.max


# In[107]:


g.sum()


# In[119]:


# to find total number of words in each PlayerLine so that to find the most important Player.

importantPlayer = df.groupby('Player')['NoOfWordsInPlayerLine'].sum()
print (importantPlayer)


# In[120]:


# converting result into dataframe

importantPlayer= importantPlayer.to_frame()


# In[121]:



importantPlayer


# In[123]:


importantPlayer['Player'] = importantPlayer.index.tolist()


# In[124]:


importantPlayer


# In[125]:


importantPlayer.index = np.arange(0,len(importantPlayer))


# In[126]:


importantPlayer


# In[129]:


importantPlayer.columns =['NoOfWordsInPlayerLine','Player']


# In[130]:


importantPlayer


# In[131]:


importantPlayer.sort_values('NoOfWordsInPlayerLine')


# In[132]:


df


# In[134]:


importantPlayer.sort_values(by='NoOfWordsInPlayerLine', ascending=False)


# In[135]:


df.to_csv('Shakespeare_ds_numberOfWordsCol.csv')


# In[136]:


importantPlayer = importantPlayer.reset_index(drop=True)


# In[138]:


importantPlayer=importantPlayer.sort_values(by='NoOfWordsInPlayerLine', ascending=False)


# In[139]:


importantPlayer


# In[140]:


importantPlayer.index = np.arange(0,len(importantPlayer))


# In[141]:


importantPlayer


# In[144]:


plt.figure(figsize=(100,100))
ax= sns.barplot(x='NoOfWordsInPlayerLine',y='Player',data=importantPlayer)
ax.set(xlabel='NoOfWordsInPlayerLine', ylabel='Player')
plt.show()

