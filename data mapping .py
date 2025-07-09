# Data mapping :

import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

# importing csv file data
orders=pd.read_csv("C:\\dataset\\final project data\\archive (21)\\orders.csv")
prod=pd.read_csv("C:\Guvi\Final_project\products.csv")
dept=pd.read_csv("C:\Guvi\Final_project\departments.csv")

aisles=pd.read_csv("C:/dataset/final project data/archive (21)/aisles.csv",delimiter=',')
aisles_new=pd.DataFrame(aisles)
aisles_new.head()

order_prod_prior=pd.read_csv("C:\Guvi\Final_project\order_products__prior.csv")
order_prod_train=pd.read_csv("C:\Guvi\Final_project\order_products__train.csv")

order_prod_train.head()
order_prod_prior.head()

# checking the avaiable column names
orders.columns
order_prod_prior.columns

# merge order and order product prior based on order id 
order_prod_pr=pd.merge(orders,order_prod_prior,how='inner',on='order_id')
order_prod_pr.head()
# merge product data into order_prod_pr 
df=pd.merge(order_prod_pr,prod,how='inner',on='product_id')
df.shape
df.head()

# merge product data into df table
df=pd.merge(df,dept,how='inner',on='department_id')

# merge aisle table
df=pd.merge(df,aisles,how='inner',on='aisle_id')

# export the data in csv

df.to_csv('INSTAMART.csv',index=False)




