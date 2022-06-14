## Importing the required libraries
import pandas as pd
import matplotlib.pyplot as plt


# ## Data  Preprocessing and Visualization

# In[159]:


## Reading a csv file in pandas data frame

df = pd.read_csv("data.csv")


# In[160]:


## first 5 values
df.head()


# In[161]:


## reading data  till the year 2021 and accumulated accidents 
df1= df[(df.Year<=2021) & (df.Month=='Summe')]


# In[162]:


df1.head()


# In[163]:


## Accumulating accidents by year
df2 = df1.groupby('Year', )['Value'].sum().to_dict() 


# In[164]:


df2 = pd.DataFrame(list(df2.items()),columns = ['year','value']) 


# In[165]:


df2.head()


# In[166]:


# create data
x = df2["year"]
y = df2["value"]


# In[167]:


# plot total accidents by year
plt.bar(x, y)
plt.xlabel("Year")
plt.ylabel("Total accidents")
plt.show()


#  2001 and 2019 has the most accidents

# In[168]:


df3 = df1.groupby(['Year', 'Category'])['Value'].nlargest(5)


# In[169]:


df3 = df1.groupby(['Year', 'Category'])['Value'].sum()


# In[170]:


type(df3)


# In[171]:


df3


# In[172]:


df1.groupby(['Year', 'Category'])['Value'].sum().unstack().plot(kind='bar',stacked = True)


# In[173]:


plt.show()


# Maximum accidents are through Verkehrsunfalle category

# In[180]:


## Data preprocessing for target value prediction

df= df[(df.Year<=2021) & (df.Category=='Alkoholunfälle') & (df.AccidentType=='insgesamt') & (df.Month!='Summe')]


# In[181]:


df


# In[182]:


df.head(30)


# In[183]:


df['Month'] = df['Month'].str[4:]


# In[184]:


df


# In[190]:


# check if there are any missing values
df.isnull().sum()


# In[188]:


df.groupby(['Year', 'Month'])['Value'].sum().unstack().plot(kind='bar',stacked = True)


# In[189]:


plt.show()


# Data is now ready for model development!

# In[ ]:




