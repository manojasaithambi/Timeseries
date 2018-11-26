
# coding: utf-8

# In[162]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Bidirectional, Dropout
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from numpy.random import seed


# In[163]:


os.listdir()


# In[164]:


df = pd.read_csv('olist_public_dataset_v2.csv', parse_dates=['order_purchase_timestamp', 'order_aproved_at',                                                             'order_estimated_delivery_date','order_delivered_customer_date',                                                              'review_creation_date','review_answer_timestamp'])


# In[165]:


print(df.shape)
df.head()


# In[166]:


df.info()


# In[167]:


#Null value verification
df.isna().sum()


# In[168]:


#2 features review_comment_title and review_comment_message have majority of NULL values. Lets drop them
df = df.drop(labels=['review_comment_title','review_comment_message'],axis=1)


# In[169]:


#There are 2405 orders for which customer deliver date is Null. Let's find the order status for these records
df[df['order_delivered_customer_date'].isna()].groupby(by=['order_status']).size()


# In[170]:


#Based on the order status these orders have not reached the customer yet. 
#Let's ignore these records for forecasting the products_value
df.dropna(subset=['order_delivered_customer_date'],inplace=True)
print(df.shape)


# In[171]:


#So now there should be records only for order_status 'delivered'. But there are few cancelled orders. 
#The orders might got delivered to customer and then cancelled or might be erroneous data. Lets ignore them too

print(df.groupby(by='order_status').size())
df = df[df['order_status']!='canceled']
print(df.shape)


# In[172]:


#Let's check for columns which have only unique values
df.nunique()


# In[173]:


#Lets remove order_status field alone
df = df.drop(labels=['order_status'],axis=1)


# In[174]:


#Let's analyze date fields
for fields in df.columns:
    if df[fields].dtype == 'datetime64[ns]':
        #df[fields] = df[fields].apply(lambda x: x.date())
        print('Range of values for',fields,' ', df[fields].min(), ',',df[fields].max())


# In[175]:


#Little bit of feature engineering.Converting few datetime fields to days (integer value)
#order estimated days - Estimated No.of days for product delivery from date of purchase.
#order delivered days - Actual No.of days for product delivery
#review answer days - No.of days it took to respond for a review

df['order_estimated_days']= (df['order_estimated_delivery_date']-df['order_purchase_timestamp']).dt.days
df['order_delivered_days'] = (df['order_delivered_customer_date']-df['order_purchase_timestamp']).dt.days
df['review_answer_days'] = (df['review_answer_timestamp']-df['review_creation_date']).dt.days


# In[176]:


#Dropping other date fields now

df = df.drop(labels=['order_aproved_at','order_estimated_delivery_date','order_delivered_customer_date','review_creation_date','review_answer_timestamp'],axis=1)
print(df.shape)


# In[177]:


#We are going to forecast order_products_value based on order_purchase_timestamp. 
#Lets ignore the id columns which doesnt help out our forecasting

df = df.drop(labels=['order_id','customer_id','product_id','review_id'],axis=1)
print(df.shape)


# In[178]:


df.head()


# In[179]:


#Lets visuzalize some categorical variables
fig, ax =plt.subplots(nrows=2,ncols=2,figsize=(15,8))
city_df = df.groupby(by=['customer_city'])['order_products_value'].size().sort_values(ascending=False).reset_index().head(20)
sns.barplot(x='customer_city',y='order_products_value',data=city_df,ax=ax[0,0])
plt.xticks(rotation=90);

zip_df = df.groupby(by=['customer_zip_code_prefix'])['order_products_value'].size().sort_values(ascending=False).reset_index().head(20)
sns.barplot(x='customer_zip_code_prefix',y='order_products_value',data=zip_df,ax=ax[0,1])
plt.xticks(rotation=90);

state_df = df.groupby(by=['customer_state'])['order_products_value'].size().sort_values(ascending=False).reset_index().head(20)
sns.barplot(x='customer_state',y='order_products_value',data=state_df,ax=ax[1,0])
plt.xticks(rotation=90);

product_df = df.groupby(by=['product_category_name'])['order_products_value'].size().sort_values(ascending=False).reset_index().head(20)
sns.barplot(x='product_category_name',y='order_products_value',data=product_df,ax=ax[1,1])
plt.xticks(rotation=90);


# In[180]:


#customer_city is the lower granular level. Lets keep that and remove other geo features
df = df.drop(labels=['customer_state','customer_zip_code_prefix'],axis=1)
print(df.shape)


# In[181]:


#With customer city and product_category_name model overfits. Based on experiments dropping the below fields
#df = df.drop(labels=['product_category_name'],axis=1)
df = df.drop(labels=['customer_city'],axis=1)


# In[182]:


df.info()


# In[183]:


df = pd.get_dummies(df)
df.shape


# In[184]:


df.set_index('order_purchase_timestamp',inplace=True)
df.index=[x.date() for x in df.index]


# In[185]:


group_df = df.groupby(df.index).sum()


# In[186]:


plt.figure(figsize=(15,8))
group_df['order_products_value'].plot()
plt.xticks(rotation=90)


# In[187]:


#There are some missing values. Need to derive a logic to find index from where no missing values are available

group_df.index=pd.to_datetime(group_df.index)
group_df['lag_index']=group_df.index
group_df['lag_index']=group_df['lag_index'].shift(-1)
group_df['lag_index'].fillna(method='ffill',inplace=True)
group_df['lag_days']=(group_df['lag_index']-group_df.index).dt.days
group_df.head(15)


# In[188]:


#group_df.corr()
group_df['row_freq']=[sum(group_df['lag_days'][i:])/len(group_df['lag_days'][i:]) for i in group_df.index]
group_df = group_df[group_df['row_freq']<=1]
group_df.head()


# In[189]:


#group_df = group_df.loc['2017-01-04':]


# In[190]:


#group_df.head()


# In[191]:


group_df = group_df.drop(labels=['lag_index','lag_days','row_freq'],axis=1)


# In[192]:


group_df.head()


# In[193]:


#print(group_df.resample('W').sum().head())
#print(group_df.resample('W').mean().head())
group_df = group_df.resample('W').mean()
plt.figure(figsize=(15,8))
group_df['order_products_value'].plot()
plt.xticks(rotation=90)


# In[194]:


actual_collength = len(group_df.columns)
scaler = MinMaxScaler()
group_df1 = scaler.fit_transform(group_df.values)
group_df1 = pd.DataFrame(group_df1,columns=group_df.columns)


# In[195]:


#Function to create timestep lag records. No_lags parameter holds the timesteps
def create_lags(data, no_lags):
    columns = data.columns
    for i in range(1, no_lags+1):
        for col in columns:
            data[col+'+'+str(i)] = data[col].shift(-i)
    return data


# In[196]:


lag_value=3
lagdata = create_lags(group_df1.copy(),lag_value)
lagdata.head()


# In[197]:


def prep_traindata(data,tgt,lag_value):
    y_data = data[tgt+'+'+str(lag_value)]
    X_data = data.iloc[:,:group_df1.shape[1]*lag_value]
    X_data.dropna(inplace=True)
    y_data.dropna(inplace=True)
    test_data = X_data.iloc[-1,:]
    X_data = X_data.iloc[:-1,:]
    return X_data, y_data, test_data
    


# In[198]:


X_data, y_data, test_data = prep_traindata(lagdata, 'order_products_value',lag_value)
print(X_data.shape, y_data.shape)
X_values = X_data.values
y_values = y_data.values
test_values = test_data.values


# In[199]:


def train_test_split(X_values, y_values, prop):
    train_size = int(X_values.shape[0]*prop)
    train_X, train_y = X_values[:train_size,:], y_values[:train_size,:]
    val_X, val_y = X_values[train_size:,:], y_values[train_size:,:]
    return train_X, train_y, val_X, val_y


# In[200]:


y_values = y_values.reshape((y_values.shape[0],1))
train_X, train_y, val_X, val_y = train_test_split(X_values, y_values, 0.8)
print(train_X.shape, train_y.shape, val_X.shape, val_y.shape)


# In[201]:


train_X = train_X.reshape((train_X.shape[0],lag_value,int(train_X.shape[1]/lag_value)))
val_X = val_X.reshape((val_X.shape[0],lag_value,int(val_X.shape[1]/lag_value)))
test_values = test_values.reshape((test_values.shape[0],-1))
test_values = test_values.reshape((test_values.shape[1],lag_value,int(test_values.shape[0]/lag_value)))
print(train_X.shape, val_X.shape, test_values.shape)


# In[202]:


seed(5)
model = Sequential()
model.add(Bidirectional(LSTM(50, kernel_initializer='lecun_normal',recurrent_dropout=0.2,input_shape=(train_X.shape[1], train_X.shape[2]))))
#model.add(Dropout(0.20))
#model.add(Bidirectional(LSTM(30, kernel_initializer='lecun_normal', recurrent_dropout=0.2)))
#model.add(Dropout(0.20))
#model.add(Dense(12,activation='relu'))
#model.add(Dropout(0.20))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')

lrcallback = ReduceLROnPlateau(factor=0.5, patience=5, verbose=1, min_lr=1E-10)
#checkpoint = ModelCheckpoint(filepath, monitor='val_loss',verbose=0,save_best_only=True,mode='min')
history = model.fit(train_X, train_y, epochs=50, batch_size=4, validation_data=(val_X, val_y), verbose=1, shuffle=False,
                   callbacks=[lrcallback])
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


# In[203]:


# Get val_loss value from the checkpoint model name
def get_loss(names):
    basename = os.path.splitext(names)[0]
    loss = basename.split('-')[1]
    return loss

pred = scaler.inverse_transform(np.concatenate((model.predict(val_X), val_X.reshape((val_X.shape[0], -1))), axis=1)[:,:actual_collength])[:,0]

actual = scaler.inverse_transform(np.concatenate((val_y, val_X.reshape((val_X.shape[0], -1))), axis=1)[:,:actual_collength])[:,0]

train_pred = scaler.inverse_transform(np.concatenate((model.predict(train_X), train_X.reshape((train_X.shape[0], -1))), axis=1)[:,:actual_collength])[:,0]

train_actual = scaler.inverse_transform(np.concatenate((train_y, train_X.reshape((train_X.shape[0], -1))), axis=1)[:,:actual_collength])[:,0]

avalues = np.concatenate((train_actual,actual))
pvalues = np.concatenate((train_pred,pred))


forecast = scaler.inverse_transform(np.concatenate((model.predict(test_values), test_values.reshape((test_values.shape[0], -1))), axis=1)[:,:actual_collength])[:,0]

yearsFmt = mdates.DateFormatter('%Y-%m-%d')
next_week =(group_df.index[-1]+datetime.timedelta(days=7)).strftime ('%Y-%m-%d')

print('Order products value for next week',next_week,'is: ',scaler.inverse_transform(np.concatenate((model.predict(test_values), test_values.reshape((test_values.shape[0], -1))), axis=1)[:,:actual_collength])[:,0])
print()
print('RMSE score: ', np.sqrt(mean_squared_error(avalues,pvalues)))

mean_forecast = [np.mean(avalues)]*len(avalues)
mean_forecast1 = [np.mean(avalues)]*len(actual)

print('RMSE score for mean forecast: ', np.sqrt(mean_squared_error(avalues,mean_forecast)))

print('RMSE score for validation set: ', np.sqrt(mean_squared_error(actual,pred)))
print('RMSE score for validation set with mean forecast: ', np.sqrt(mean_squared_error(actual,mean_forecast1)))

fig,ax1 = plt.subplots(figsize=(15,8))
ax1.plot(group_df.index[lag_value:], np.concatenate((train_actual,actual)), label='actual_output',marker='o')
ax1.plot(group_df.index[lag_value:], np.concatenate((train_pred,pred)), label='lstm_output',marker='o')
ax1.plot(group_df.index[lag_value:], mean_forecast, label='mean_forecast',color='blue')
ax1.scatter(x=next_week,y=forecast, label='forecast_output', color='green')
ax1.xaxis.set_major_formatter(yearsFmt)
plt.xticks(rotation=90)
plt.xlabel('Week date')
plt.ylabel('Order Products value')
plt.title('RMSE score: {:.4f}'.format(np.sqrt(mean_squared_error(actual,pred))))
plt.legend()
plt.savefig('prediction_plot.jpg')
plt.show()

