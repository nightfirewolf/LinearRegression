import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


rf = RandomForestRegressor(max_depth=2, random_state=100)

lr =LinearRegression()
df = pd.read_csv('file:///C:/Users/AUGUSTINE%20DEINNE/Downloads/book2.csv')


y = df['expenses']
x = df.drop('expenses', axis=1)


X_train, X_test, Y_train,Y_test = train_test_split(x,y, test_size=0.2, random_state=100)
lr.fit(X_train,Y_train)



y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)



lr_train_mse = mean_squared_error(Y_train,y_lr_train_pred)
lr_train_r2 = r2_score(Y_train,y_lr_train_pred)

lr_test_mse = mean_squared_error(Y_test,y_lr_test_pred)
lr_test_r2 = r2_score(Y_test,y_lr_test_pred)

 

lr_results = pd.DataFrame(['Linear reqression', lr_train_mse,lr_train_r2,lr_test_mse,lr_test_r2]).transpose()
lr_results.columns = ['Method', 'Training Mse', 'Training R2', 'Test MSE ', 'Test R2']




X_train, X_test, Y_train,Y_test = train_test_split(x,y, test_size=0.2, random_state=100)
rf.fit(X_train,Y_train)



y_rf_train_pred = rf.predict(X_train)
y_rf_test_pred = rf.predict(X_test)



rf_train_mse = mean_squared_error(Y_train,y_rf_train_pred)
rf_train_r2 = r2_score(Y_train,y_rf_train_pred)

rf_test_mse = mean_squared_error(Y_test,y_rf_test_pred)
rf_test_r2 = r2_score(Y_test,y_rf_test_pred)

 

rf_results = pd.DataFrame(['Linear reqression', rf_train_mse,rf_train_r2,rf_test_mse,rf_test_r2]).transpose()
rf_results.columns = ['Method', 'Training Mse', 'Training R2', 'Test MSE ', 'Test R2']


df_models = pd.concat([rf_results,rf_results], axis=0)
print(df_models.reset_index(drop=True))


plt.figure(figsize=(5,5))
plt.scatter(x=Y_train, y=y_lr_train_pred, c='#7CAE00',alpha=0.3)

z = np.polyfit(Y_train, y_lr_train_pred,1)
p = np.poly1d(z)

plt.plot(Y_train, p(Y_train), '#F8766D')
plt.ylabel('Predict logs')
plt.xlabel('Experiment Logs')





