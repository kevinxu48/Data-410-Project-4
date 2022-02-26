# Multivariate Regression Analysis and Gradient Boosting

## Multivariate Regression Analysis
Multivariate regression is a regression technique that estimates a regression model with one dependent/response variable, while using more than one independent/predictor variable in a multivariate regression model, the model is a multivariate multiple regression. 

In general for Multivariate models, for <img src="https://render.githubusercontent.com/render/math?math=n"> number of features, we want 

<img src="https://render.githubusercontent.com/render/math?math=E(y | X_1,X_2,\cdots, X_n) = F(X_1,X_2,\cdots, X_n) = Y">

where F is the model or regressor we consider.

One of the most common type of multivariate Regression is multiple linear regression, which fits a linear equation to given data in an attempt to model the relationship between a single independent variable and a dependent variable, with any other features having weights of 0. The fitting of the data points becomes harder to visualize as more features are included in the model, but we can visualize the fitting of data in lower dimensions.

*Example of 1-dimensional linear regression*

![multivariate reg](https://user-images.githubusercontent.com/98488236/155559992-a3eaf59e-dff7-442b-a75c-b0515f9c1422.gif)

*Example of 2-dimensional linear regression*

![2d](https://user-images.githubusercontent.com/98488236/155585904-ef005af3-1488-47ba-99e9-c81c1d647c5d.png)

Notice for a model with two independent variables, a plane is fitted to the data points rather than a line.

## Feature Selection
One important factor in multivariate regression is choosing the features that are the most important. There can often be multicolinearity between certain independent variables, so removing the irrelevant or redundant features will improve learning accuracy on the training data and reduces the overall run time. Often, the most **parsimonious** model is selected with the least number of explanatory variables possible while still explaining the data.

In this project, we will use **Principle Component Analysis (PCA)** to determine the three most important features when there are multiple to consider. PCA is a dimensionality reduction technique used to obtain a subset of features based on the direction of the biggest variablity. by projecting (dot product) the original data into the reduced PCA space using the eigenvectors of the correlation matrix, which are known as the principal components, and results in linear combinations of the original data that summarize most of the variablility.

*Example of PCA in 2 dimensions*

<img src="https://user-images.githubusercontent.com/98488236/155625613-5d0e4223-62af-4058-8f61-558e7741339e.PNG" width=40% height=40%>

To actually perform the feature selection, we multiply the <img src="https://render.githubusercontent.com/render/math?math=n"> variables in the selected model by a binary weight vector <img src="https://render.githubusercontent.com/render/math?math=(w_1,w_2,\cdots, w_n)">, <img src="https://render.githubusercontent.com/render/math?math=w_p\in \{0,1\}"> which has the value 1 if a variable should be included in the model or a value of 0 otherwise. This vector of weights is known as the **sparsity pattern** or the incidence matrix.

Given the sparsity pattern, the mathematical function of a Multivariate regression model is usually of the form: 

![mlr model](https://user-images.githubusercontent.com/98488236/155565976-21670880-71ec-46f3-a823-44cdadcf699a.jpg)

 where <img src="https://render.githubusercontent.com/render/math?math=\large\beta_i"> determines the contribution of the independent variable <img src="https://render.githubusercontent.com/render/math?math=\large Y = X_i"> and <img src="https://render.githubusercontent.com/render/math?math=\large\sigma"> is the residual standard deviation with the residuals being i.i.d. with an expected value of 0. The sparsity pattern means that a subset of the <img src="https://render.githubusercontent.com/render/math?math=\beta">'s will be 0. 
 
## Rank Deficiency
We can solve for the weights by utilizing some Algebra and the method of Least Squares to obtain:

![equation](https://user-images.githubusercontent.com/98488236/155569885-3253811f-acd1-4f05-92f1-e6befd4f0c6b.jpg)

However, we reach a problem when <img src="https://render.githubusercontent.com/render/math?math=X^TX"> is not invertible, which can happen when you have too little data such that there are more columns than rows in X or if there is strong multicolinearity between some of the independent variables; this is known as **rank deficiency**. 

*Example of rank deficient data*

![rank deficient](https://user-images.githubusercontent.com/98488236/155574702-22481663-87ae-4e87-968e-3fa7b34a0dc4.jpg)


To bypass this issue we can take the <img src="https://render.githubusercontent.com/render/math?math=X^TX"> matrix and add the identity matrix scaled by some small value <img src="https://render.githubusercontent.com/render/math?math=\large\lambda">. In the case of multicolinearity, we can use **Regularization** techniques, such as the L1 and L2 regularization models to minimize the sum of the square residuals.
### Applications of Multivariate Regression
Here are some regression problems that we can solve using Multivariate Regression (we will be doing the second and third in this project):
- Predict the median value of a house given variables such as average number of rooms, amount per capita crime, proportions of non-retail business acres per town, etc.
- Predict the miles per gallon of a car given variables such as its weight, number of cylinder, and engine displacement
- Predict the strength of concrete given variables such as the amount of concrete, fly ash, water, etc. present in the mixture, and the age in days 
### Advantages of Multivariate Regression
- We are able to find relationships between the dependent variable and the independent variables, specifically we can find the relative influence of the predictor variables to the target variable
- Since there are multiple variables to consider, the conclusions drawn tend to be more accurate and better represent real-life situations
- Multivariate analysis may reduce the likelihood of making Type I errors (rejecting a true null hypothesis).
### Disadvantages of Multivariate Regression
- The process of feature selection leaves the possibility of lurking variables, which is when the relationship between two variables is significantly affected by the presence of a third variable which has not been included in the modeling.
- Unsuitable for small data sets because of the problems of rank deficiency

# Gradient Boosting for Regression
Gradient Boosting can be used for both classification and regression problems, but for this project we will mainly focus on its regression applications. The concept of Gradient boosting originated with the idea of turning a *weak learner*, a learner whose performance is marginally better than random guessing, into a stronger learner. Moreover, it combines the concept of Gradient Descent and "Boosting", such that an additive model in the form of a Decision Tree is added to a weak learner to optimize a loss function, in our case we will minimize the Mean Squared Error on the test sets through the addition of our Decision Tree.

In general, say we are given a weak learner such as a regressor <img src="https://render.githubusercontent.com/render/math?math=\large F"> that for <img src="https://render.githubusercontent.com/render/math?math=\large n"> observations makes predictions: <img src="https://render.githubusercontent.com/render/math?math=\large F(x_i)"> for the ith observation for <img src="https://render.githubusercontent.com/render/math?math=i \in \{1,2,\cdots, n\}">. To turn <img src="https://render.githubusercontent.com/render/math?math=\large F"> into a stronger learner, a decision tree <img src="https://render.githubusercontent.com/render/math?math=\large h"> is trained with the goal of predicting the residuals with the outputs being the residuals <img src="https://render.githubusercontent.com/render/math?math=\large y_i - F(x_i)"> which are the difference between the observed and predicted values.

To start the algorithm, a single leaf is created that represents the initial guess for the weights of all the samples. Gradient Boost's first prediction is the average value of the dependent variable, and a Decision Tree is built from this prediction that has a restricted number of leaves. The algorithm will continue to build trees based on the errors of previous trees, and only stops until additional trees do not improve the fit or it reaches the max number of trees you will set as a hyperparameter.

*Description of how residuals are calculated and used to build the trees*
![gradientboosting](https://user-images.githubusercontent.com/98488236/155253952-775739d0-fd0a-4e1a-a228-553192934fee.png)

To prevent overfitting and reduce variance, each tree's contribution to the prediction can be scaled with a learning rate between 0 and 1.

## Extreme Gradient Boosting (XGB)
XGB is an open source implementation of Gradient Boosting that improves upon the base Gradient Boosting framework through optimizations such as changing the order of loops to improve run time, using a max_depth parameter to improve computational performance, handling sparsity patterns more efficiently, and prevens overfitting through Lasso (L1) and Ridge (L2) regularization. Hence, it is known for being faster, more memory efficient and accurate compared to other implementations of Gradient Boosting. 

XGBoost has multiple hyperparameters that we can tune: the objective/learning task which is MSE in this project, the number of gradient boosted trees, the maximum depth of the trees, Gamma-the minimum loss reduction required to partition a leaf, Lambda and Alpha- the L1 and L2 norms respectively that control the weights, and the learning rate. To implement XGB in python, we can simply install the xgboost package by running:

```
import xgboost as xgb
```
### Advantages of XGB
- It is known for performing regression and classification tasks very quickly
- It performs especially well on structured datasets with not too many features. 
- It is a robust algorithm that prevents over-fitting

### Disadvantages of XGB
- It does not work as well on unstructured data
- It is sensitive to outliers, since boosting methods build each tree on the previous trees' residuals
# Comparison of Various Regression Methods
## New Kernels for Lowess
In addition to the tricubic, quartic, and Epanechnikov kernels we implemented in project 2, this project will implement two additional kernels: Triweight and Cosine.
```
# Triweight Kernel
def Triweight(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,35/32*(1-d**2)**3)
```

```
# Plot of the Triweight Kernel
x = np.arange(-1.5,1.51,0.01)
y = Triweight(x)
fig, ax = plt.subplots()
ax.plot(x,y,lw=2)
plt.show()
```

![triweight](https://user-images.githubusercontent.com/98488236/155627828-f0f18879-e24f-4972-a867-a0f77934e64d.png)

```
# Cosine Kernel
def Cosine(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,np.pi/4*(np.cos(np.pi/2 *d)))
```
```
# Plot of the Cosine Kernel
x = np.arange(-1.5,1.51,0.01)
y = Cosine(x)
fig, ax = plt.subplots()
ax.plot(x,y,lw=2)
plt.show()
```
![cosine](https://user-images.githubusercontent.com/98488236/155627892-368920c3-1858-4144-9ffb-ef954d2e56a5.png)

The other kernels are the same as described in Project 2.
## Implementation of Lowess and Boosted Lowess Regression
Since Project 2, we have added an intercept parameter to the lowess function and also created a boosted lowess function.
```
#Defining the kernel local regression model

def lw_reg(X, y, xnew, kern, tau, intercept):
    # tau is called bandwidth K((x-x[i])/(2*tau))
    n = len(X) # the number of observations
    yest = np.zeros(n)

    if len(y.shape)==1: # here we make column vectors
      y = y.reshape(-1,1)

    if len(X.shape)==1:
      X = X.reshape(-1,1)
    
    if intercept:
      X1 = np.column_stack([np.ones((len(X),1)),X])
    else:
      X1 = X

    w = np.array([kern((X - X[i])/(2*tau)) for i in range(n)]) # here we compute n vectors of weights

    #Looping through all X-points
    for i in range(n):          
        W = np.diag(w[:,i])
        b = np.transpose(X1).dot(W).dot(y)
        A = np.transpose(X1).dot(W).dot(X1)
        #A = A + 0.001*np.eye(X1.shape[1]) # if we want L2 regularization
        #theta = linalg.solve(A, b) # A*theta = b
        beta, res, rnk, s = lstsq(A, b)
        yest[i] = np.dot(X1[i],beta)
    if X.shape[1]==1:
      f = interp1d(X.flatten(),yest,fill_value='extrapolate')
    else:
      f = LinearNDInterpolator(X, yest)
    output = f(xnew) # the output may have NaN's where the data points from xnew are outside the convex hull of X
    if sum(np.isnan(output))>0:
      g = NearestNDInterpolator(X,y.ravel()) 
      # output[np.isnan(output)] = g(X[np.isnan(output)])
      output[np.isnan(output)] = g(xnew[np.isnan(output)])
    return output
```
```
# boosted Lowess model

def boosted_lwr(X, y, xnew, kern, tau, intercept):
  # we need decision trees
  # for training the boosted method we use X and y
  Fx = lw_reg(X,y,X,kern,tau,intercept) # we need this for training the Decision Tree
  # Now train the Decision Tree on y_i - F(x_i)
  new_y = y - Fx
  #model = DecisionTreeRegressor(max_depth=2, random_state=123)
  model = RFR(n_estimators=100,max_depth=2)
  #model = model_xgb
  model.fit(X,new_y)
  output = model.predict(xnew) + lw_reg(X,y,xnew,kern,tau,intercept)
  return output 
```
## Data Sets
In this project we will test the regression methods on the cars and a new concrete strength data set. For the cars dataset, we will use all 3 features: number of cylinder, and engine displacement, and weight to predict the Miles Per Gallon. 

```
# Obtain and scale the features for the car dataset
scale = SS()

Xcars = cars[['CYL', 'ENG', 'WGT']].values
ycars = cars['MPG'].values

Xcars_scaled = scale.fit_transform(Xcars)
```

The concrete dataset is more complicated in that it has 8 features: age of the mixture and the amount of Cement, Blast Furnace Slag, Fly Ash, Water, Superplasticizer, Coarse Aggregate, and Fine Aggregate in the mixture to predict the concrete strength. Using PCA we will narrow it down to the 3 features that summarize most of the variability.
```
X_conc = concrete[['cement','slag','ash', 'water', 'superplastic', 'coarseagg', 'fineagg', 'age']].values
y_conc = concrete[['strength']].values

Xconc_scale = scale.fit_transform(X_conc)
```
```
pca = PCA(n_components=3)
Xconc_new = pca.fit_transform(Xconc_scale) # project the original data into the PCA space
print(abs( pca.components_))
```

[[0.09840137 0.17726197 0.39466185 0.54700395 0.50594541 0.03792808
  0.40192597 0.29147949]
 [0.11373709 0.6860529  0.14294751 0.05325628 0.2829296  0.62994342
  0.01939111 0.12598089]
 [0.81420224 0.17179437 0.40822055 0.21318975 0.23459653 0.17408781
  0.00456921 0.10052137]]
  
 From the PCA, we see that variables 4 (water), 2 (slag), and 1 (concrete) summarize most of the variability, so we will select those as our features.

## K-Fold validations
To perform the K-Fold validations, two functions were created: one for the boosted and unboosted Lowess, and one for the other models. We perform nested KFolds using a range of random states to better validate the results.
```
def DoNestedKFoldLoess(x,y,k, kern, tau, boosted, intercept):
  scale = SS()
  mse_lwr = []
  for i in range(10):
    kf = KFold(n_splits=k,shuffle=True,random_state=i)
    for idxtrain, idxtest in kf.split(x):
      xtrain = x[idxtrain]
      ytrain = y[idxtrain]
      ytest = y[idxtest]
      xtest = x[idxtest]
      xtrain = scale.fit_transform(xtrain)
      xtest = scale.transform(xtest)

      # call the boosted lowess function if it is desired
      if boosted:  
        yhat_blwr = boosted_lwr(xtrain,ytrain, xtest,kern,tau,intercept)
        mse_lwr.append(mse(ytest,yhat_blwr))
      else:
        yhat_lwr = lw_reg(xtrain, ytrain,xtest,kern,tau, intercept)
        mse_lwr.append(mse(ytest,yhat_lwr))
  return np.mean(mse_lwr)
```
### Cars dataset
Now using the cars data, we will perform nested 10-Fold validations for Lowess, Boosted Lowess, XGBoost, Random Forest, and the Nadaraya–Watson kernel regression techniques and comapre the results.

### Finding the optimal kernel for Lowess and Boosted Lowess
To find the best kernel to use for Lowess, we performed nested 10-Fold validations to determine which kernel produced the lowest MSE values for Lowess and boosted Lowess. 

```
# obtain a list of Lowess MSE values for each kernel
mse_lwr = []
for kern in [Epanechnikov, Tricubic, Triweight, Cosine, Quartic]:
  mse_blwr.append(DoNestedKFoldLoess(Xcars, ycars, 10, kern, 0.9, False, True))
mse_lwr
```
[16.907313732004027,
 16.91047562708034,
 16.893423122247928,
 16.904756187972154,
 16.89986707278945]
 
```
# obtain a list of boosted Lowess MSE values for each kernel
mse_blwr = []
for kern in [Epanechnikov, Tricubic, Triweight, Cosine, Quartic]:
  mse_blwr.append(DoNestedKFoldLoess(Xcars, ycars, 10, kern, 0.9, True, True))
mse_blwr
```
[16.66542963597202,
 16.666600952115058,
 16.69434012649239,
 16.664779472237093,
 16.716134961065322]
 
From the K-Fold validation results, we see that the Triweight kernel obtained the best results for Lowess Regression, but the Cosine performed the best for boosted Lowess. Hence, when tuning the hyperparameters to perform comparisons, we will use Triweight kernel for Lowess and a Cosine kernel for Boosted Lowess.

## Tuning the hyperparameters
### Lowess and Boosted Lowess
To tune the hyperparameters for Lowess and Boosted Lowess for the cars dataset, we performed non-nested K-Fold validations for a certain tau range for both Lowess and Boosted Lowess using their respective optimal kernels.

```
# Narrow a good value for the tau parameter for Lowess with triweight kern
k = 10
t_range = np.arange(0.9,3, step=.1)
test_mse = []
for t in t_range:
  te_mse = DoKFoldLoess(Xcars,ycars,k,Triweight, t, False, True)
  test_mse.append(np.mean(te_mse))
```

```
# plot the test mse to find the best value for tau

idx = np.argmin(test_mse)
print([t_range[idx], test_mse[idx]])
plt.plot(t_range,test_mse, '-xr',label='Test')
plt.xlabel('tau value')
plt.ylabel('Avg. MSE')
plt.title('K-fold validation with k = ' + str(k))
plt.show()
```

[1.2999999999999998, 17.06326606831356]

<img src="https://user-images.githubusercontent.com/98488236/155764542-02e92cc8-1536-4f84-86d7-0f5e7d00eed1.png" width=40% height=40%>

```
# Narrow a good value for the tau parameter for Boosted Lowess with triweight kern
k = 10
t_range = np.arange(0.1,1, step=.1)
test_mse = []
for t in t_range:
  te_mse = DoKFoldLoess(Xcars,ycars,k,Cosine, t, True, True)
  test_mse.append(np.mean(te_mse))
```

```
# plot the test mse to find the best value for tau

idx = np.argmin(test_mse)
print([t_range[idx], test_mse[idx]])
plt.plot(t_range,test_mse, '-xr',label='Test')
plt.xlabel('tau value')
plt.ylabel('Avg. MSE')
plt.title('K-fold validation with k = ' + str(k))
plt.show()
```

[0.8, 16.94224157210573]

<img src="https://user-images.githubusercontent.com/98488236/155766277-2e33e7a8-7c8b-4e5d-9224-fe2702edd7ee.png" width=40% height=40%>

Hence, from the K-Fold validations, we get an optimized tau of 1.3 for Lowess Regression and a tau of 0.8 for Boosted Lowess Regression.

### Random Forest Regression
For Random Forest Regression, we will be tuning the max_depth and n_estimators parameters. From plotting results for a max depth from 1-5, we have determined that a max_depth of 3 produces the best cross-validated results when paired with other hyperparameters. Hence, we will tune the number of trees based on a max_depth of 3.

```
# First use K-Fold to narrow a good value for the n_estimator
k = 10
t_range = np.arange(60,200, step=1)
test_mse = []
for t in t_range:
  rfr = RFR(n_estimators = t, max_depth = 3, random_state=410)
  te_mse = DoKFold(rfr,Xcars,ycars,k,410)
  test_mse.append(np.mean(te_mse))
```

```

# plot the test mse to find the best value for n_estimators

idx = np.argmin(test_mse)
print([t_range[idx], test_mse[idx]])
plt.plot(t_range,test_mse, '-xr',label='Test')
plt.xlabel('number of trees')
plt.ylabel('Avg. MSE')
plt.title('K-fold validation with k = ' + str(k))
plt.show()
```
[178, 16.961528636028113]

<img src="https://user-images.githubusercontent.com/98488236/155767913-9b31cf3c-b07e-434f-b24a-8c78aca6ab15.png" width=40% height=40%>

From the results of the cross-validations we get that the optimal hyperparameter values are 178 trees with a maximum depth of 3.

### XGBoost
There are many hyperparameters to consider for XGBoost, so it is difficult to determine exactly what the best value is for each parameter. Spoiler Alert: it appears that regardless of the values chosen for the hyperparameters, XGBoost performs leagues better than the other regression techniques. Hence, we did a basic iteration through different values for hyperparameters and found that the parameters that obtained the best cross-validated results was:   
```
 model_xgb = xgb.XGBRegressor(objective='reg:squarederror',n_estimators=60,reg_lambda=15,alpha=0,gamma=12.4,max_depth=2)
```

This is likely not the most optimal combination of the hyperparameters, but regardless it will serve as an efficient model.

## Results for Cars Dataset
We compared the nested cross-validated results for Lowess, Boosted Lowess, RandomForest, and XGBoost regressors, and we obtained these results.

```
model_rf = RFR(n_estimators = 178, max_depth=3)

print('The Cross-validated MSE for LWR is : ' +str(DoNestedKFoldLoess(Xcars, ycars, 10, Triweight, 1.3, False, True)))
print('The Cross-validated MSE for BLWR is : '+str(DoNestedKFoldLoess(Xcars, ycars,10,  Cosine, 0.8, True, True)))
print('The Cross-validated MSE for RF is : '+str(DoKFold(model_rf,Xcars,ycars,10,410,True)))
print('The Cross-validated MSE for XGB is : '+str(DoKFoldXGB(Xcars,ycars, 'reg:squarederror',60,15,0,12.4,2, 410, True)))
```
*Results from the Nested K-Fold validations*

The Cross-validated MSE for LWR is : 16.959251056181547
The Cross-validated MSE for BLWR is : 16.687318388260138
The Cross-validated MSE for RF is : 16.80957591130398
The Cross-validated MSE for XGB is : 15.731945730590105

From the MSE values, we clearly see that XGBoost performed far better than any other technique, and that Boosted Lowess performed marginally better than Random Forest, and Lowess Regression came last.

### Concrete dataset
Now we will repeat our steps on the concrete strength dataset to verify that the results were not just a fluke.

## Finding the Best Kernel for Lowess and Boosted Lowesss
Here we repeat the same process as for the cars dataset, except instead of performing nested K-Fold validations, we will simply perform standard cross-validations because the concrete dataset is much larger, and results in a significantly longer runtime.
```
mse_blwr = []
for kern in [Epanechnikov, Tricubic, Triweight, Cosine, Quartic]:
  mse_blwr.append(DoKFoldLoess(X_conc, y_conc, 10, kern, 0.9, False, True))
  mse_blwr
```

[147.50254368228087,
 146.4227649591804,
 145.88023526561545,
 147.2636028932561,
 146.34599111816914]
 
 ```
 mse_blwr = []
for kern in [Epanechnikov, Tricubic, Triweight, Cosine, Quartic]:
  mse_blwr.append(DoKFoldLoess(X_conc, y_conc, 10, kern, 0.9, True, True))
  mse_blwr
 ```
 
 [142.5072230143699,
 142.58612792373117,
 142.74523440367054,
 142.44087542013045,
 142.13448284090876]
 
We see that similarly, a Triweight kernel produced the best results for Lowess, but a Quartic kernel produced the best results for Boosted Lowess. Perhaps the lack of nested loops resulted in a random state that was more favorable for a Quartic kernel, but regardless we will proceed with a Triweight kernel for Lowess regression and a Quartic kernel for boosted Lowess regression.

## Tuning the hyperparameters
### Lowess and Boosted Lowess
To tune the hyperparameters for Lowess and Boosted Lowess for the concrete strength dataset, we performed similar non-nested K-Fold validations for a certain tau range for both Lowess and Boosted Lowess using their respective optimal kernels.

```
# Narrow a good value for the tau parameter for Lowess with triweight kern
k = 10
t_range = np.arange(0.1,3, step=.1)
test_mse = []
for t in t_range:
  te_mse = DoKFoldLoess(X_conc,y_conc,k,Triweight, t, False, True)
  test_mse.append(np.mean(te_mse))
# plot the test mse to find the best value for tau
```

```
idx = np.argmin(test_mse)
print([t_range[idx], test_mse[idx]])
plt.plot(t_range,test_mse, '-xr',label='Test')
plt.xlabel('tau value')
plt.ylabel('Avg. MSE')
plt.title('K-fold validation with k = ' + str(k))
plt.show()
```

[0.9, 145.88023526561545]

<img src=https://user-images.githubusercontent.com/98488236/155820074-19aecf09-fea8-4421-afaf-fded220899c2.png width=40% height=40%>


```
# Narrow a good value for the tau parameter for Lowess with Quartic kern
k = 10
t_range = np.arange(0.1,3, step=.1)
test_mse = []
for t in t_range:
  te_mse = DoKFoldLoess(X_conc,y_conc,k,Quartic, t, False, True)
  test_mse.append(np.mean(te_mse))
# plot the test mse to find the best value for tau
```

```
idx = np.argmin(test_mse)
print([t_range[idx], test_mse[idx]])
plt.plot(t_range,test_mse, '-xr',label='Test')
plt.xlabel('tau value')
plt.ylabel('Avg. MSE')
plt.title('K-fold validation with k = ' + str(k))
plt.show()
```

[1.3000000000000003, 142.0938341529481]

<img src=https://user-images.githubusercontent.com/98488236/155823893-05d9507a-02c6-4a3e-b902-6cc17e6511fb.png width=40% height=40%>

From the results obtained, we will use a tau value of 0.9 for Lowess and a tau value of 1.3 for Boosted Lowess.

### Random Forest Regressor
Like before for Random Forest Regression, we will be tuning the max_depth and n_estimators parameters. This time from plotting results for a max depth from 1-7, we have determined that a max_depth of 6 produces the best cross-validated results. Hence, we will tune the number of trees based on a max_depth of 6.

[59, 146.53884274619216]

<img src=(https://user-images.githubusercontent.com/98488236/155825157-eb716883-c788-4f57-a524-a223e786828c.png width=40% height=40%>

From the results of the cross-validations we get that the optimal hyperparameter values are 59 trees with a maximum depth of 6.

### XGBoost
Like before, we did basic iterations to tune each hyperparameter, and got that a 
found that the parameters that obtained the best cross-validated results was:   
```
 model_xgb = xgb.XGBRegressor(objective='reg:squarederror',n_estimators=60,reg_lambda=15,alpha=0,gamma=12.4,max_depth=2)
```



# Conclusion

## References
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3049417/

https://brilliant.org/wiki/multivariate-regression/#:~:text=Multivariate%20Regression%20is%20a%20method,responses)%2C%20are%20linearly%20related.&text=A%20mathematical%20model%2C%20based%20on,and%20other%20more%20complicated%20questions.

https://towardsdatascience.com/applied-multivariate-regression-faef8ddbf807

https://towardsdatascience.com/graphs-and-ml-multiple-linear-regression-c6920a1f2e70

https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/

https://towardsdatascience.com/understanding-adaboost-2f94f22d5bfe

https://towardsdatascience.com/xgboost-python-example-42777d01001e

https://towardsdatascience.com/machine-learning-part-18-boosting-algorithms-gradient-boosting-in-python-ef5ae6965be4

https://www.geeksforgeeks.org/ml-gradient-boosting/

https://en.wikipedia.org/wiki/Kernel_(statistics)

https://towardsdatascience.com/https-medium-com-vishalmorde-xgboost-algorithm-long-she-may-rein-edd9f99be63d
