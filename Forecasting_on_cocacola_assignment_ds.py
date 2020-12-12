#!/usr/bin/env python
# coding: utf-8

# # Importing the required libraries

# In[1]:


import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import seaborn as sns


# # Loading the Dataset

# In[2]:


cocacola = pd.read_excel(r"C:\Users\Binita Mandal\Desktop\finity\forecasting\CocaCola_Sales_Rawdata.xlsx")


# In[3]:


#finding first rows
cocacola.head()


# In[4]:


#Finding last rows
cocacola.tail()


# In[5]:


# Lets see the shape and size of dataset
cocacola.shape


# In[6]:


cocacola.size


# In[7]:


# Information about the dataset
cocacola.info()


# In[9]:


cocacola.describe()


# In[10]:


cocacola


# In[11]:


# PLotting the data
cocacola.Sales.plot()


# In[12]:


cocacola['Quarters']= 0
cocacola['Year'] = 0
for i in range(42):
    p = cocacola["Quarter"][i]
    cocacola['Quarters'][i]= p[0:2]
    cocacola['Year'][i]= p[3:5]


# In[13]:


# Prepring dummies 
Quarters_Dummies = pd.DataFrame(pd.get_dummies(cocacola['Quarters']))
cocacola1 = pd.concat([cocacola,Quarters_Dummies],axis = 1)


# In[14]:


cocacola1["t"]=np.arange(1,43)


# In[15]:


cocacola1["t_squared"] = cocacola1["t"]*cocacola1["t"]
cocacola1.columns
cocacola1["Log_Sales"] = np.log(cocacola1["Sales"])


# ### Heat map

# In[16]:


plt.figure(figsize=(12,8))
heatmap_y_month = pd.pivot_table(data=cocacola,values="Sales",index="Year",columns="Quarters",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g") #fmt is format of the grid values


# In[17]:


sns.boxplot(x="Quarters",y="Sales",data=cocacola1)
sns.boxplot(x="Year",y="Sales",data=cocacola1)


# # Lineplot

# In[18]:


plt.figure(figsize=(12,3))
sns.lineplot(x="Year",y="Sales",data=cocacola)


# In[19]:


decompose_ts_add = seasonal_decompose(cocacola.Sales,period=12)
decompose_ts_add.plot()
plt.show()


# # Splitting data

# In[20]:


Train = cocacola1.head(38)
Test = cocacola1.tail(4)


# In[21]:


# Linear model
import statsmodels.formula.api as smf
linear_model = smf.ols('Sales~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
rmse_linear


# In[22]:


# Exponential
Exp = smf.ols('Log_Sales~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp


# In[23]:


# Quadratic
Quad = smf.ols('Sales~t+t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad


# In[24]:


# Additive seasonality
add_sea = smf.ols('Sales~Q1+Q2+Q3',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Q1','Q2','Q3']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea


# In[25]:


# Additive Seasonality Quadratic
add_sea_Quad = smf.ols('Sales~t+t_squared+Q1+Q2+Q3',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Q1','Q2','Q3','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 


# In[26]:


# Multiplicative Seasonality
Mul_sea = smf.ols('Log_Sales~Q1+Q2+Q3',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea


# In[27]:


# Multiplicative Additive Seasonality
Mul_Add_sea = smf.ols('Log_Sales~t+Q1+Q2+Q3',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 


# In[28]:


#tabulating the rmse values

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse


# #### Predict for new time period

# In[29]:


predict_data = pd.read_excel(r"C:\Users\Binita Mandal\Desktop\finity\forecasting\CocaCola_New.xlsx")


# In[30]:


predict_data


# In[31]:


#Build the model on entire data set
model_full = smf.ols('Sales~t',data=cocacola1).fit()
pred_new  = pd.Series(model_full.predict(predict_data))
pred_new


# In[32]:


predict_data["forecasted_Sales"] = pd.Series(pred_new)
predict_data


# #### Conclusion:- for this method we can say that Multiplicative Additive Seasonality is the best fit model.

# In[ ]:




