# Ex-07-Feature-Selection

# AIM

To Perform the various feature selection techniques on a dataset and save the data to a file.

# Explanation
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.

# ALGORITHM
STEP 1

Read the given Data

STEP 2

Clean the Data Set using Data Cleaning Process

STEP 3

Apply Feature selection techniques to all the features of the data set

STEP 4

Save the data to the file

# CODE :
```
Developed By: D.Vishnu vardhan reddy
Reg.No: 212221230023
```
```
#loading dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("titanic_dataset.csv")
df

#checking data
df.isnull().sum()


# removing unnecessary data variables
df.drop('Cabin',axis=1,inplace=True)
df.drop('Name',axis=1,inplace=True)
df.drop('Ticket',axis=1,inplace=True)
df.drop('PassengerId',axis=1,inplace=True)
df.drop('Parch',axis=1,inplace=True)
#cleaning data
df['Age']=df['Age'].fillna(df['Age'].median())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
df.isnull().sum()



#removing outliers 
plt.title("Dataset with outliers")
df.boxplot()
plt.show()

cols = ['Age','SibSp','Fare']
Q1 = df[cols].quantile(0.25)
Q3 = df[cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
plt.title("Dataset after removing outliers")
df.boxplot()
plt.show()


#feature encoding 
from sklearn.preprocessing import OrdinalEncoder
embark=["C","S","Q"]
emb=OrdinalEncoder(categories=[embark])
df["Embarked"]=emb.fit_transform(df[["Embarked"]])

from category_encoders import BinaryEncoder
be=BinaryEncoder()
df["Sex"]=be.fit_transform(df[["Sex"]])
df


#feature scaling
from sklearn.preprocessing import RobustScaler
sc=RobustScaler()
df2=pd.DataFrame(sc.fit_transform(df),columns=['Survived','Pclass','Sex','Age','SibSp','Fare','Embarked'])
df2


#feature transformation
df2.skew()

import statsmodels.api as sm
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal',n_quantiles=692)
#no skew data that need no transformation- Age, Embarked
#moderately positive skew- Survived, Sex
#highy positive skew- Fare, SibSp
#highy negative skew- Pclass
df1=pd.DataFrame()
df1["Survived"]=np.sqrt(df2["Survived"])
df1["Pclass"],parameters=stats.yeojohnson(df2["Pclass"])
df1["Sex"]=np.sqrt(df2["Sex"])
df1["Age"]=df2["Age"]
df1["SibSp"],parameters=stats.yeojohnson(df2["SibSp"])
df1["Fare"],parameters=stats.yeojohnson(df2["Fare"])
df1["Embarked"]=df2["Embarked"]
df1.skew()


#feature selection process
import matplotlib
import seaborn as sns
import statsmodels.api as sm
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

X = df1.drop("Survived",1) 
y = df1["Survived"]          

plt.figure(figsize=(12,10))
cor = df1.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.RdPu)
plt.show()

cor_target = abs(cor["Survived"])
relevant_features = cor_target[cor_target>0.5]
relevant_features

X_1 = sm.add_constant(X)
model = sm.OLS(y,X_1).fit()
model.pvalues


#Backward Elimination
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)

model = LinearRegression()
#Initializing RFE model
rfe = RFE(model, 4)
#Transforming data using RFE
X_rfe = rfe.fit_transform(X,y)  
#Fitting the data to model
model.fit(X_rfe,y)
print(rfe.support_)
print(rfe.ranking_)

nof_list=np.arange(1,6)            
high_score=0
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))

cols = list(X.columns)
model = LinearRegression()
rfe = RFE(model, 2)             
X_rfe = rfe.fit_transform(X,y)  
model.fit(X_rfe,y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)



#Embedded Method
reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
plt.show()
```
# OUTPUT :
![a](https://user-images.githubusercontent.com/94175324/170305947-ed78bd08-68d2-433d-afcd-150af5dd79c7.png)
![a1](https://user-images.githubusercontent.com/94175324/170306031-966c8b32-5dea-497d-aefd-1cef80b91dab.png)
![a3](https://user-images.githubusercontent.com/94175324/170306161-62c26cff-d01f-4f33-b1b0-7fd748d3d661.png)
![a4](https://user-images.githubusercontent.com/94175324/170306195-91003f58-8808-442b-8a83-817f87c3212f.png)
![a5](https://user-images.githubusercontent.com/94175324/170306229-bdb2529f-98b1-4a69-80ff-b299b0f4c8c8.png)
![a6](https://user-images.githubusercontent.com/94175324/170306263-f4bca011-681e-4479-bc87-6de9f5e32366.png)
![a7](https://user-images.githubusercontent.com/94175324/170306290-daa9cd8d-cead-4db6-8670-a0b77cb4a085.png)
![a8](https://user-images.githubusercontent.com/94175324/170306317-dcbc47cc-e938-47b9-a77a-f1444a6c17fd.png)
![a9](https://user-images.githubusercontent.com/94175324/170306350-ad79ad56-4263-424b-bf37-9503c9ecded2.png)
![S1](https://user-images.githubusercontent.com/94175324/170307856-581c7899-ff6f-4567-b3e9-e08b75d0cd8e.png)
![a11](https://user-images.githubusercontent.com/94175324/170306409-91985d24-194a-48ff-b064-6a5a58ada7d9.png)
![S2](https://user-images.githubusercontent.com/94175324/170307763-68ee9854-8dac-4bde-a1da-468c368a8467.png)
![a13](https://user-images.githubusercontent.com/94175324/170306471-36c2c751-5b08-4c70-8310-7d6254614831.png)
![a14](https://user-images.githubusercontent.com/94175324/170306522-8382737d-2b7a-4ead-9f6f-2d50412d1645.png)


# RESULT :
Thus, the various feature selection techniques have been performed on a given dataset successfully.












