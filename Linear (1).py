#!/usr/bin/env python
# coding: utf-8

# In[86]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import warnings 
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[87]:


df = pd.read_csv("house_prices.csv")
df.head()


# In[4]:


df.info()


# In[54]:


df.isnull().sum()/len(df)*100


# In[6]:


df.drop(columns=["Plot Area","Dimensions","Super Area","Ownership","Society","overlooking","facing"],inplace=True)


# In[7]:


df["Carpet Area"] = df["Carpet Area"].str.replace(" sqft", "",regex=False)
df["Carpet Area"] = pd.to_numeric(df["Carpet Area"],errors="coerce")


# In[8]:


df["Status"].fillna("Unknown",inplace=True)


# In[9]:


df["Description"].fillna("Unknown",inplace=True)


# In[10]:


df["Transaction"].fillna("Unknown",inplace=True)


# In[11]:


df["Bathroom"] = pd.to_numeric(df["Bathroom"],errors='coerce')
df["Bathroom"].fillna(df["Bathroom"].median(),inplace=True)


# In[12]:


df.head


# In[30]:


plt.hist(df["Carpet Area"].dropna())
plt.show()


# In[31]:


plt.boxplot(df["Carpet Area"].dropna())
plt.show()


# In[32]:


Q1 = df["Carpet Area"].quantile(0.25)
Q3 = df["Carpet Area"].quantile(0.75)
IQR = Q3-Q1

lower = Q1-1.5*IQR
upper = Q3+1.5*IQR

outliers = df[(df["Carpet Area"]<lower) | (df["Carpet Area"]>upper)].index
print(outliers)


# In[33]:


df["Carpet Area"] = np.where(df["Carpet Area"]<lower,lower,np.where(df["Carpet Area"]>upper,upper,df["Carpet Area"]))


# In[34]:


df["Carpet Area"].skew()


# In[35]:


df["Carpet Area"].fillna(df["Carpet Area"].median(),inplace=True)


# In[36]:


print(outliers)


# In[37]:


df["Floor"] = df["Floor"].astype(str).str.extract(r'(\d+)')
df["Floor"] = pd.to_numeric(df["Floor"],errors="coerce")


# In[38]:


df.head()


# In[39]:


plt.boxplot(df["Floor"].dropna())
plt.show()


# In[40]:


Q1 = df["Floor"].quantile(0.25)
Q3 = df["Floor"].quantile(0.75)
IQR = Q3-Q1

lower = Q1-1.5*IQR
upper = Q3+1.5*IQR

outliers = df[(df["Floor"]<lower) | (df["Floor"]>upper)].index
print(outliers)


# In[41]:


df["Floor"] = np.where(df["Floor"]<lower,lower,np.where(df["Floor"]>upper,upper,df["Floor"]))


# In[42]:


plt.hist(df["Floor"].dropna())
plt.show()


# In[43]:


df["Floor"].skew()


# In[44]:


df["Floor"].fillna(df["Floor"].median(),inplace=True)


# In[45]:


df["Furnishing"].fillna("Unknown",inplace=True)


# In[53]:


df["Balcony"].fillna(df["Balcony"].median(),inplace=True)


# In[47]:


df.tail()


# In[48]:


df["Balcony"] = pd.to_numeric(df["Balcony"],errors='coerce')
df["Balcony"].fillna(df["Balcony"].median(),inplace=True)


# In[49]:


df["Price (in rupees)"].fillna(df["Price (in rupees)"].median(),inplace=True)


# In[50]:


df.columns


# In[51]:


df["Car Parking"].fillna("Unknown",inplace=True)


# In[52]:


df.drop(columns=["Car Parking"],inplace=True)


# In[58]:


def amount_convert(x):
    x= x.strip()
    if "Lac" in x:
        return float(x.replace("Lac", "").strip())*1e5
    elif "Cr" in x:
        return float(x.replace("Cr", "").strip())*1e7
    else:
        return None

df["Amount(in rupees)"] = df["Amount(in rupees)"].apply(amount_convert)




# In[63]:


df.head()


# In[62]:


# feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaling= ["Amount(in rupees)","Price (in rupees)","Carpet Area","Floor","Bathroom","Balcony"] 
df[scaling]= scaler.fit_transform(df[scaling])


# In[83]:


#testing_training
from sklearn.model_selection import train_test_split
X =  df.select_dtypes(include=['int64', 'float64']).drop(columns=["Amount(in rupees)","Price (in rupees)"])
Y = df["Amount(in rupees)"]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=24)


# In[84]:


#model trainig
model = LinearRegression()


# In[90]:


model.fit(X_train,Y_train)
predictions = model.predict(X_test)
# Evaluate
print("R² Score:", r2_score(Y_test, prediction))
print("MSE:", mean_squared_error(Y_test, prediction))


# In[80]:


df.isnull().sum()/len(df)*100


# In[81]:


df.head()


# In[82]:


df["Amount(in rupees)"].fillna(df["Amount(in rupees)"].median(),inplace=True)


# In[94]:


plt.scatter(X_train["Carpet Area"], Y_train, color="blue", label="Train Data")
plt.scatter(X_test["Carpet Area"], Y_test, color="green", label="Test Data")
plt.plot(X_test["Carpet Area"], prediction, color="red", linewidth=2, label="Prediction Line")

plt.xlabel("Carpet Area")
plt.ylabel("Amount (in rupees)")
plt.legend()
plt.show()


# In[ ]:




