
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from subprocess import check_output
get_ipython().run_line_magic('matplotlib', 'inline')


# In[122]:


df=pd.read_csv('C:\\Users\\p860n111\\Desktop\\data science\\Hw2\\Shakespeare_data.csv')


# In[4]:


df.describe


# In[5]:


#to check the values for columns that have NaN value in them 
print(df.info())


# In[6]:


df.head(10)


# In[7]:


df.describe()


# In[8]:


# to see the total number of columns in the dataset.
df.columns


# In[9]:


#to find the total number of unique play
print("Number of plays are: " + str(df['Play'].nunique()))


# In[123]:


# to replace the NaN value in the Player column to be Unknown
df['Player'].replace(np.nan, 'Unknown',inplace = True)


# In[10]:


# to check that the Player column has all i.e. 111396 non null values.
df.info()


# In[12]:


# to verify that the missing values in Palyer column got replaced by Unknown
df.head()


# In[13]:


# to print the count of unique plays and their names
print("The total number of plays are: " + str(df['Play'].nunique())+" and their names are as follows: " + str(df['Play'].unique()))


# In[14]:


#creating a new data frame that will contain unique Plays value as a new column named: Name of the Play

pd.DataFrame(df['Play'].unique().tolist(), columns=['Name of the Play'])


# In[15]:


# Additional Information 1: For each Play, number of lines (PlayerLine) spoken by each PlaYer

df.groupby(['Play','Player']).count()['PlayerLine']


# In[16]:


# Now converting the above data into a frame (playWise_lines_per_player).

playWise_lines_per_player= df.groupby(['Play','Player']).count()['PlayerLine']
playWise_lines_per_player= playWise_lines_per_player.to_frame()
playWise_lines_per_player


# In[17]:


playWise_lines_per_player.describe()


# In[18]:


# Additional Information 2: To count the number of PlayerLine corresponding to each Play.

df.groupby('Play').count().sort_values(by='PlayerLine',ascending=True)['PlayerLine']


# In[19]:


#Converting the data into a dataframe (playerLinePerPlay)
playerLinePerPlay = df.groupby('Play').count().sort_values(by='PlayerLine',ascending=True)['PlayerLine']


# In[20]:


playerLinePerPlay


# In[21]:


playerLinePerPlay = playerLinePerPlay.to_frame()


# In[22]:


playerLinePerPlay


# In[23]:


# applying indexing to the above dataframe
playerLinePerPlay['Play'] = playerLinePerPlay.index.tolist()


# In[24]:


playerLinePerPlay


# In[25]:


playerLinePerPlay.index = np.arange(0,len(playerLinePerPlay))


# In[26]:


playerLinePerPlay


# In[27]:


# plotting a graph to show: PlayerLine against Name of the Play
plt.figure(figsize=(90,50))
ax= sns.barplot(x='Play',y='PlayerLine',data=playerLinePerPlay, order = playerLinePerPlay['Play'])
ax.set(xlabel='Name of the Play', ylabel='PlayerLines')
plt.show()


# In[28]:


plt.figure(figsize=(15,15))
ax= sns.barplot(x='PlayerLine',y='Play',data=playerLinePerPlay, order = playerLinePerPlay['Play'])
ax.set(xlabel='PlayerLines', ylabel='Name of the Play')
plt.show()


# In[30]:


# Additional Information 3: Number of Players corresponding to each Play

playersPerPlay = df.groupby(['Play'])['Player'].nunique().sort_values(ascending= True)


# In[31]:


playersPerPlay


# In[32]:


#changing to it to dataframe
playersPerPlay=playersPerPlay.to_frame()


# In[33]:


playersPerPlay


# In[34]:


playersPerPlay['Play'] = playersPerPlay.index.tolist()


# In[35]:


playersPerPlay


# In[36]:


# now to change the index from Play to 0 - (length-1) adn renaming the column name 

playersPerPlay.columns = ['Number of Players','Name of the Play']


# In[37]:


playersPerPlay


# In[38]:


playersPerPlay.index= np.arange(0,len(playersPerPlay))
playersPerPlay


# In[39]:


# plotting graph

plt.figure(figsize=(15,15))
ax = sns.barplot(x='Number of Players',y='Name of the Play',data=playersPerPlay)
ax.set(xlabel='Number of Players', ylabel='Name of the Play')
plt.show()


# In[40]:


plt.figure(figsize=(100,100))
ax = sns.barplot(x='Name of the Play',y='Number of Players',data=playersPerPlay)
ax.set(xlabel='Name of the Play', ylabel='Number of Players')
plt.show()


# In[42]:


# to calculate the total words in each PlayerLine cell entry
df['new_column'] = df.PlayerLine.apply(lambda x: len(str(x).split(' ')))


# In[43]:


df


# In[44]:


df.groupby(['Player'])['new_column']


# In[45]:


df


# In[46]:


df


# In[50]:



df.rename(columns={'new_column': 'NoOfWordsInPlayerLine'}, inplace=True)


# In[51]:


df


# In[52]:


h=df.groupby('Player')


# In[53]:


for Player, data in h:
    print("Player:",Player)
    print("\n")
    print("Player:",data)


# In[54]:


h


# In[54]:


g.max


# In[56]:


g.sum()


# In[57]:


# to find total number of words in each PlayerLine so that to find the most important Player.

importantPlayer = df.groupby('Player')['NoOfWordsInPlayerLine'].sum()
print (importantPlayer)


# In[58]:


# converting result into dataframe

importantPlayer= importantPlayer.to_frame()


# In[59]:



importantPlayer


# In[60]:


importantPlayer['Player'] = importantPlayer.index.tolist()


# In[61]:


importantPlayer


# In[62]:


importantPlayer.index = np.arange(0,len(importantPlayer))


# In[63]:


importantPlayer


# In[64]:


importantPlayer.columns =['NoOfWordsInPlayerLine','Player']


# In[65]:


importantPlayer


# In[66]:


importantPlayer.sort_values('NoOfWordsInPlayerLine')


# In[67]:


df


# In[68]:


importantPlayer.sort_values(by='NoOfWordsInPlayerLine', ascending=False)


# In[69]:


df.to_csv('Shakespeare_ds_numberOfWordsCol.csv')


# In[70]:


importantPlayer = importantPlayer.reset_index(drop=True)


# In[71]:


importantPlayer=importantPlayer.sort_values(by='NoOfWordsInPlayerLine', ascending=False)


# In[72]:


importantPlayer


# In[73]:


importantPlayer.index = np.arange(0,len(importantPlayer))


# In[74]:


importantPlayer


# In[75]:


plt.figure(figsize=(100,100))
ax= sns.barplot(x='NoOfWordsInPlayerLine',y='Player',data=importantPlayer)
ax.set(xlabel='NoOfWordsInPlayerLine', ylabel='Player')
plt.show()


# In[76]:


importantPlayer


# In[77]:


importantPlayer.to_csv('Shakespeare_ds_importantPlayer.csv')


# In[78]:


df


# In[ ]:


#df_uniqueWords_Count_In_PlayerLine=pd.DataFrame(r1,columns=['PlayerLine'])


# In[76]:


df


# In[79]:


from collections import Counter
result = Counter(" ".join(df['PlayerLine'].values.tolist()).split(" ")).items()
result


# In[79]:


from collections import Counter
result = Counter(" ".join(df['PlayerLine'].values.tolist()).lower().split(" ")).items()
result


# In[80]:


most_Common_Word_df = pd.DataFrame([result])


# In[81]:


most_Common_Word_df


# In[82]:


most_Common_Word_df.to_csv('Shakespeare_ds_Most_common_word.csv')


# In[105]:


from pandas import ExcelWriter

writer = ExcelWriter('Shakespeare_ds_Most_common_word.xlsx')
most_Common_Word_df.to_excel(writer,'Sheet5')
writer.save()


# In[83]:


play_name = df['Play'].unique().tolist()
for play in play_name:
    p_line = df[df['Play']==play].groupby('Player').count().sort_values(by='PlayerLine',ascending=True)['PlayerLine']
    p_line = p_line.to_frame()
    p_line['Player'] = p_line.index.tolist()
    p_line.index = np.arange(0,len(p_line))
    p_line.columns=['Lines','Player']
    plt.figure(figsize=(10,10))
    ax= sns.barplot(x='Lines',y='Player',data=p_line)
    ax.set(xlabel='Number of Lines', ylabel='Player')
    plt.title(play,fontsize=30)
    plt.show()


# In[110]:


play_name = df['Play'].unique().tolist()
for play in play_name:
    p_line = df[df['Play']==play].groupby('Player').count().sort_values(by='PlayerLine',ascending=False)['PlayerLine']
    p_line = p_line.to_frame()
    p_line['Player'] = p_line.index.tolist()
    p_line.index = np.arange(0,len(p_line))
    p_line.columns=['Lines','Player']
    plt.figure(figsize=(10,10))
    ax= sns.barplot(x='Lines',y='Player',data=p_line)
    ax.set(xlabel='Number of Lines', ylabel='Player')
    plt.title(play,fontsize=30)
    plt.show()


# In[114]:


g= nx.Graph()


# In[116]:


df


# In[134]:


trainingDS=df.sample(frac=0.8,random_state=200)
testingDS=df.drop(train.index)


# In[135]:



trainingDS


# In[84]:


testingDS.describe()


# In[138]:


testingDS.info()


# In[ ]:


testingDS.info()


# In[140]:


from sklearn.model_selection import train_test_split
trainingSet, testSet = train_test_split(df, test_size=0.2)


# In[142]:


trainingSet.info()


# In[ ]:


testSet.info()


# In[144]:


trainingSet["Player"].value_counts().plot(kind="bar")
trainingSet["Player"].value_counts()


# In[221]:



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df.PlayerLinenumber, df.Play, test_size=0.25, random_state=0)


# In[218]:



from sklearn.linear_model import LogisticRegression


# In[219]:


logisticRegr = LogisticRegression()


# In[223]:


logisticRegr.fit(x_train), y_train)


# In[96]:


df


# In[131]:


df.drop('ActSceneLine', axis=1, inplace=True)


# In[132]:


df


# In[111]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df.Player, df.Play, test_size=0.25, random_state=0)


# In[157]:


X = df.loc[:, df.columns != 'Player']
y = df.loc[:, df.columns == 'Play']


# In[158]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[141]:


X


# In[136]:


y


# In[142]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


# In[159]:


df=pd.read_csv('C:\\Users\\p860n111\\Desktop\\data science\\Hw2\\Shakespeare_data.csv')


# In[160]:


df = df.dropna()
print(df.shape)
print(list(df.columns))


# In[154]:


X = df.iloc[:,1:]
y = df.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[155]:


classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)


# In[152]:


df.drop('PlayerLine', axis=1, inplace=True)


# In[156]:


df


# In[172]:


df = pd.read_csv('C:\\Users\\p860n111\\Desktop\\data science\\Hw2\\Shakespeare_data.csv')


# In[173]:


df.head()


# In[174]:


df = df.dropna()
print(df.shape)
print(list(df.columns))


# In[164]:


feature_cols = ['Player', 'Play']


# In[165]:


# you want all rows, and the feature_cols' columns
X = train.loc[:, feature_cols]


# In[168]:


# now we want to create our response vector
y = train.Player


# In[169]:


# 1. import
from sklearn.linear_model import LogisticRegression

# 2. instantiate model
logreg = LogisticRegression()

# 3. fit 
logreg.fit(X, y)


# In[175]:


df.info()


# In[176]:


df.drop('Dataline', axis=1, inplace=True)
df.drop('PlayerLinenumber', axis=1, inplace=True)
df.drop('ActSceneLine', axis=1, inplace=True)
df.drop('Dataline', axis=1, inplace=True)
df.drop('PlayerLine', axis=1, inplace=True)


# In[177]:


df.drop('Dataline', axis=1, inplace=True)


# In[179]:


df.drop('PlayerLine', axis=1, inplace=True)


# In[180]:


df


# In[186]:


x =df.ix[:,0].values
y = df.ix[:,1].values


# In[189]:


X_train, X_test, y_train, y_test = train_test_split(xogReg = LogisticRegression()
LogReg.fit(X_train, y_train), y, test_size=0.25, random_state=0)


# In[191]:


LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)


# In[194]:


X_train


# In[197]:


df.info()


# In[227]:


df = pd.read_csv('C:\\Users\\p860n111\\Desktop\\data science\\Hw2\\Shakespeare_data.csv')


# In[228]:


df = df.dropna()
print(df.shape)
print(list(df.columns))


# In[224]:


# Import train_test_split function
from sklearn.model_selection import train_test_split

X=df[['PlayerLinenumber']]  # Features
y=df['Player']  # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1) # 70% training and 30% test


# In[225]:


#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)


# In[226]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[229]:


df.info()


# In[230]:


df.dtypes


# In[231]:


obj_df = df.select_dtypes(include=['object']).copy()
obj_df.head()


# In[232]:


df["Play"] = df["Play"].astype('category')
df.dtypes


# In[233]:


df["Play_cat"] = df["Play"].cat.codes
df.head()


# In[237]:


df.dtypes


# In[236]:


df["PlayerLinenumber"] = df["PlayerLinenumber"].astype('category')
df.dtypes
df["PlayerLinenumber_cat"] = df["PlayerLinenumber"].cat.codes
df.head()


# In[238]:


df["ActSceneLine"] = df["ActSceneLine"].astype('category')
df.dtypes
df["ActSceneLine_cat"] = df["ActSceneLine"].cat.codes
df.head()


# In[241]:


df.dtypes


# In[240]:


df["Player"] = df["Player"].astype('category')
df.dtypes
df["Player_cat"] = df["Player"].cat.codes
df.head()


# In[242]:


df["PlayerLine"] = df["PlayerLine"].astype('category')
df.dtypes
df["PlayerLine_cat"] = df["PlayerLine"].cat.codes
df.head()


# In[243]:


df.dtypes


# In[245]:


newDF = df.filter(['Dataline','Play_cat','PlayerLinenumber_cat','ActSceneLine_cat','Player_cat','PlayerLine_cat' ], axis=1)


# In[248]:


newDF.dtypes


# In[249]:


X = newDF.ix[:,(0,1,2,3,5)].values
y = newDF.ix[:,4].values


# In[250]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=25)


# In[251]:


LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)


# In[253]:


predictions = LogReg.predict(X_test)


# In[254]:


LogReg.predict(X_test[0:10])


# In[255]:


predictions = LogReg.predict(X_test)


# In[256]:


# Use score method to get accuracy of model
score = LogReg.score(X_test,  y_test)
print(score)


# In[263]:


X1 = newDF.ix[:,(1,2)].values
y1 = newDF.ix[:,4].values


# In[257]:


newDF.dtypes


# In[268]:


X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size = .2, random_state=25)


# In[ ]:


LogReg = LogisticRegression()
LogReg.fit(X_train1, y_train1)


# In[ ]:


predictions = LogReg.predict(X_test1)


# In[267]:


# Use score method to get accuracy of model
score = LogReg.score(X_test1,  y_test1)
print(score)

