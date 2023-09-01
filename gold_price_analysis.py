import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


df= pd.read_csv("gold_monthly_csv.csv")
print(df.head())
print(df.tail())
print(df.shape)

print(f"Date range of gold prices available from - {df.loc[:,'Date'][0]} to {df.loc[:,'Date'][len(df)-1]}")

date=pd.date_range(start='1/1/1950',end='8/1/2020',freq='M')
print(date)
df['month']=date
df.drop('Date',axis=1,inplace=True)
df=df.set_index('month')
print(df.head())

 # Here we are doing EDA (Exploratory DAta Analysis)
df.plot(figsize=(20,8))
plt.title("Gold Prices monthly since 1950 and onwards")
plt.xlabel("months")
plt.ylabel("Prices")
plt.grid()
#plt.show()

print(round(df.describe(),3))
_, ax=plt.subplots(figsize=(25,8))
sns.boxplot(x= df.index.year,y= df.values[:,0],ax=ax)
plt.title("Gold price (monthly since 1950 onwards)")
plt.xlabel("year")
plt.ylabel("Price")
plt.xticks(rotation=90)
plt.grid()


from statsmodels.graphics.tsaplots import month_plot
fig,ax = plt.subplots(figsize=(22,8))
month_plot(df, ylabel = 'Gold Price', ax=ax)
plt.title("Gold price monthly since 1950 onwards")
plt.xlabel("month")
plt.ylabel("Price")
plt.grid();
#plt.show()
_,ax=plt.subplots(figsize=(22,8))
sns.boxplot(x=df.index.month_name(),y=df.values[:,0],ax=ax)
plt.title("Gold price (monthly since 1950 onwards)")
plt.xlabel("year")
plt.ylabel("Price")
#plt.show()

df_yearly_sum=df.resample('A').mean()
df_yearly_sum.plot();
plt.title('Avg gold price yearly since 1960')
plt.xlabel(" Year")
plt.ylabel('Price')
plt.grid()
#plt.show()

df_quaterly_sum=df.resample('Q').mean()
df_quaterly_sum.plot();
plt.title('Avg gold price Quarterly since 1960')
plt.xlabel("Quarter")
plt.ylabel('Price')
plt.grid()
#plt.show()


df_decade_sum=df.resample('10Y').mean()
df_decade_sum.plot()
plt.title('Avg gold price per decade since 1960')
plt.xlabel("Decade")
plt.ylabel('Price')
plt.grid()
#plt.show()

#coefficient of variation
df_1=df.groupby(df.index.year).mean().rename(columns={'Price':'Mean'})
df_1=df_1.merge(df.groupby(df.index.year).std().rename(columns={'Price':'Std'}),left_index=True, right_index=True)
df_1['Cov_pct']=((df_1['Std']/df_1['Mean'])*100).round(2)
print(df_1.head())

fig,ax=plt.subplots(figsize=(15,10))
df_1['Cov_pct'].plot();
plt.title('Avg gold price yearly since 1950')
plt.xlabel("Year")
plt.ylabel('CV in %')
plt.grid()
#plt.show()

train=df[df.index.year<=2015]
test=df[df.index.year>2015]

print(train.shape)
print(test.shape)
train['Price'].plot(figsize=(13,5),fontsize=15)
test['Price'].plot(figsize=(13,5),fontsize=15)
plt.grid()
plt.legend(['Training Data','Test Data'])
#plt.show()

train_time=[i+1 for i in range(len(train))]
test_time= [i+len(train)+1 for i in range(len(test))]
print(len(train_time),len(test_time))

#linear regression
LR_train= train.copy( )
LR_test= test.copy()

LR_train['time']=train_time
LR_test['time']=test_time

lr= LinearRegression()
lr.fit(LR_train[['time']],LR_train['Price'].values)
test_predictions_model1=lr.predict(LR_test[['time']])
LR_test['forcast']=test_predictions_model1

plt.figure(figsize=(14,6))
plt.plot(train['Price'],label='train')
plt.plot(test['Price'],label='test')
plt.plot(LR_test['forcast'],label='reg on time_test data')
plt.legend(loc= 'best')
plt.grid()
#plt.show()

#MAPE 
def mape(actual, pred):
    return round((np.mean(abs(actual-pred)/actual))*100,2)

mape_model1_test=mape(test['Price'].values,test_predictions_model1)                                   
print('MAPE is %3.3f' %(mape_model1_test),'%')

results=pd.DataFrame({'Test Mape(%)':[mape_model1_test]},index=['RegressionOnTime'])
print(results)

#naive forecast
Naive_train=train.copy()
Naive_test=test.copy()

Naive_test['naive']=np.asarray(train['Price'])[len(np.asarray(train['Price']))-1]
print(Naive_test['naive'].head())
plt.figure(figsize=(12,8))
plt.plot(Naive_train['Price'],label='Train')
plt.plot(test['Price'],label='Test')
plt.plot(Naive_test['naive'],label='Naive Forecast on test data')
plt.legend(loc='best')
plt.title('naive Forecasting')
plt.grid()
#plt.show()

mape_model2_test=mape(test['Price'].values,Naive_test['naive'].values)
print('For Naive forecast on the test data, mape is %3.3f'%(mape_model2_test),'%')

resultsDF_2=pd.DataFrame({'Test MAPE(%)':[mape_model2_test]},index=['NaiveModel'])
results2=pd.concat([results,resultsDF_2])
print(results2)

#final model

final_model=ExponentialSmoothing(df,trend='additive',seasonal='additive').fit(smoothing_level=0.4,smoothing_trend=0.3,smoothing_seasonal=0.6)
mape_final_model=mape(df['Price'].values,final_model.fittedvalues)
print('MAPE:', mape_final_model)

predictions=final_model.forecast(steps=len(test))
pred_df=pd.DataFrame({'lower_CI': predictions - 1.96*np.std(final_model.resid,ddof=1),
                      'prediction':predictions,
                      'upper_CI': predictions + 1.96*np.std(final_model.resid,ddof=1)})
print(pred_df.head())

axis=df.plot(label='Actual', figsize=(16,9))
pred_df['prediction'].plot(ax=axis,label='Forecast',alpha=0.5)
axis.fill_between(pred_df.index,pred_df['lower_CI'],pred_df['upper_CI'],color='m', alpha=.15)
axis.set_xlabel('year-month')
axis.set_ylabel('Price')
plt.legend(loc='best')
plt.grid()
plt.show()


                     

     
