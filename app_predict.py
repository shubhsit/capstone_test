import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler 
import streamlit as st 
import pandas as pd 
import numpy as np 
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import os
# prepare train and test dataset
# Read dataset
demand_df = pd.read_csv('/content/Daily_Demand_Forecasting_Orders.csv', sep=';')

n_train = 48
X = demand_df.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11]].values
y = demand_df.iloc[:,-1].values
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
scaler = MinMaxScaler()
trainX= scaler.fit_transform(trainX)
testX = scaler.transform(testX)



st.write("""
# Supply chain target prediction
the app predict Target order based on Brazilian logistic dataset
## by **CDS Team ** 	""")


st.sidebar.header("User input parameter")

def user_input_data():

	Weekofthemonth=st.number_input('Week of the month',min_value=1,max_value=5)
	Dayoftheweek=st.number_input(' Day of the week',min_value=1,max_value=5)
	Nonurgentorder=st.number_input('Non-urgent order')
	Urgentorder=st.number_input('Urgent order ')
	OrdertypeA=st.number_input('Order type A')
	OrdertypeB=st.number_input('Order type B')
	OrdertypeC=st.number_input('Order type C')
	Fiscalsectororders=st.number_input('Fiscal sector orders')
	trafficcontrollerorders=st.number_input('traffic controller orders ')
	Bankingorders1=st.number_input('Banking orders (1)')
	Bankingorders2=st.number_input('Banking orders (2)')
	Bankingorders3=st.number_input('Banking orders (3)')
	data={'Weekofthemonth':Weekofthemonth,
			'Dayoftheweek':Dayoftheweek,
			'Nonurgentorder':Nonurgentorder,
			'Urgentorder':Urgentorder,
			'OrdertypeA' :OrdertypeA,
			'OrdertypeB' :OrdertypeB,
			'OrdertypeC' :OrdertypeC,
			'Fiscalsectororders' :Fiscalsectororders,
			'trafficcontrollerorders' :trafficcontrollerorders,
			'Bankingorders1' :Bankingorders1,
			'Bankingorders2' :Bankingorders2,
			'Bankingorders3' :Bankingorders3
       
       }

	features=pd.DataFrame(data,index=[0])
	return features

df=user_input_data()

st.subheader('user input parameter')
st.write(df)

# iris = datasets.load_iris()
# X= iris.data
# Y= iris.target

# clf=RandomForestClassifier()
# clf.fit(X,Y)

# prediction=clf.predict(df)
# prediction_proba=clf.predict_proba(df)
df_new= scaler.transform(df)
loaded_model = joblib.load("/content/drive/MyDrive/capstone/mlp_regressor_grid_search_best_model.h5")
y_predx=loaded_model.predict(df_new)

# st.subheader('class label and there corresponding index number ')
# st.write(iris.target_names)

st.subheader('prediction ')
st.write(y_predx)