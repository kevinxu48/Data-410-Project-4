# Multivariate Regression Analysis and Gradient Boosting

## Multivariate Regression Analysis
Multivariate regression is a regression technique that estimates a regression model with one or more dependent/response variable, while using more than one independent/predictor variable in a multivariate regression model, the model is a multivariate multiple regression.

In general for Multivariate models, for <img src="https://render.githubusercontent.com/render/math?math=n"> number of features, we want 

<img src="https://render.githubusercontent.com/render/math?math=E(y|X_1,X_2,\cdots, X_n) = F(X_1,X_2,\cdots, X_n) = Y">
where F is the model or regressor we consider.

## Feature Selection
One important factor in multivariate regression is choosing the features that are the most important. There is often multicolinearity between certain independent variables, so removing the irrelevant or redundant features will improve learning accuracy on the training data and reduces the overall run time. Often, the most **parsimonious** model is selected with the least number of explanatory variables possible while still explaining the data.

To actually perform the feature selection, we multiply the <img src="https://render.githubusercontent.com/render/math?math=n"> variables in the selected model by a binary weight vector <img src="https://render.githubusercontent.com/render/math?math=(w_1,w_2,\cdots, w_n)">, <img src="https://render.githubusercontent.com/render/math?math=w_p\in \{0,1\}"> which has the value 1 if a variable should be included in the model or a value of 0 otherwise. This vector of weights is known as the **sparsity pattern** or the incidence matrix.

## Types of Multivariate Regression 
One of the most common type of multivariate Regression is multiple linear regression, which fits a linear equation to given data in an attempt to model the relationship between a single independent variable and a dependent variable, with any other features having weights of 0. 

### Possible Applications of Multivariate Regression
- One way 
### Advantages of Multivariate Regression
Multivariate analysis may reduce the likelihood of making Type I errors (rejecting a true null hypothesis).

### Disadvantages of Multivariate Regression
The process of feature selection leaves the possibility of lurking variables, which is when the relationship between two variables is significantly affected by the presence of a third variable which has not been included in the modeling. Unsuitable for small data sets because 

# Gradient Boosting for Regression
Gradient Boosting can be used for both classification and regression problems, but for this project we will mainly focus on its regression applications. The concept of Gradient boosting originated with the idea of turning a *weak learner*, a learner whose performance is marginally better than random guessing, into a stronger learner. Moreover, it combines the concept of Gradient Descent and "Boosting", such that an additive model in the form of a Decision Tree is added to a weak learner to optimize a loss function, in our case we will minimize the Mean Squared Error on the test sets through the addition of our Decision Tree.

In general, say we are given a weak learner such as a regressor <img src="https://render.githubusercontent.com/render/math?math=F"> that for <img src="https://render.githubusercontent.com/render/math?math=n"> observations makes predictions: <img src="https://render.githubusercontent.com/render/math?math=F(x_i)"> for the ith observation for <img src="https://render.githubusercontent.com/render/math?math=i \in \{1,2,\cdots, n\}">. To turn <img src="https://render.githubusercontent.com/render/math?math=F"> into a stronger learner, a decision tree <img src="https://render.githubusercontent.com/render/math?math=h"> is trained with the goal of predicting the residuals with the outputs being the residuals <img src="https://render.githubusercontent.com/render/math?math=y_i - F(x_i)"> which are the difference between the observed and predicted values.

To start the algorithm, a single leaf is created that represents the initial guess for the weights of all the samples. Gradient Boost's first prediction is the average value of the dependent variable, and a Decision Tree is built from this prediction that has a restricted number of leaves. The algorithm will continue to build trees based on the errors of previous trees, and only stops until additional trees do not improve the fit or it reaches the max number of trees you will set as a hyperparameter.

*Description of how residuals are calculated and used to build the trees*
![gradientboosting](https://user-images.githubusercontent.com/98488236/155253952-775739d0-fd0a-4e1a-a228-553192934fee.png)


To prevent overfitting and reduce variance, each tree's contribution to the prediction is scaled with a learning rate between 0 and 1
## Extreme Gradient Boosting (XGB)
XGB is an implementation of Gradient Boosting in which regularization parameters are , and is faster, more memory efficient and accurate compared to other implementations of Gradient Boosting.
def tricubic(x):
  if len(x.shape) == 1:  # use a conditional to reshape if it is a row vector
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,70/81*(1-d**3)**3)
```
```
# Plot of the Tricubic Kernel
y = tricubic(x)
fig, ax = plt.subplots()
ax.plot(x,y,lw=2)
plt.show()
```
![tricubic](https://user-images.githubusercontent.com/98488236/152915484-7139fe6c-6e27-4add-a391-deb20cdf4af7.png)
```
# Epanechnikov kernel
def Epanechnikov(x):
  if len(x.shape) == 1:  
    x = x.reshape(-1,1)
  return np.where(np.abs(x)>1,0,3/4*(1-np.abs(x)**2)) 
```
```
# Plot of the Epanechnikov Kernel
y = Epanechnikov(x)
fig, ax = plt.subplots()
ax.plot(x,y,lw=2)
plt.show()
```
![epanechnikov](https://user-images.githubusercontent.com/98488236/152915711-8851bf57-bec8-4368-bfaa-346df657da06.png)


```
# Plot of the Quartic Kernel
# Quartic kernel
def Quartic(x):
  if len(x.shape) == 1:  
    x = x.reshape(-1,1)
  return np.where(np.abs(x)>1,0,15/16*(1-np.abs(x)**2)**2) 
```
```
y = Quartic(x)
fig, ax = plt.subplots()
ax.plot(x,y,lw=2)
plt.show()
```
![quartic](https://user-images.githubusercontent.com/98488236/152915763-a6cf2c21-96e8-44ff-9e27-f0d9568cab31.png)

### Implementation of the Lowess Regression
Python and SKlearn do not have an implementation of Lowess Regression, so to use it we must define our own function.
```
def lowess_reg(x, y, xnew, kern, tau):
    # We expect x to the sorted increasingly
    n = len(x)
    yest = np.zeros(n)

    #Initializing all weights from the bell shape kernel function    
    w = np.array([kern((x - x[i])/(2*tau)) for i in range(n)])     
    
    #Looping through all x-points
    for i in range(n):
        weights = w[:, i]
        b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
        A = np.array([[np.sum(weights), np.sum(weights * x)],
                    [np.sum(weights * x), np.sum(weights * x * x)]])
        #theta = linalg.solve(A, b) # A*theta = b
        theta, res, rnk, s = linalg.lstsq(A, b)
        yest[i] = theta[0] + theta[1] * x[i] 
    f = interp1d(x, yest,fill_value='extrapolate')
    return f(xnew)
```

# Random Forest Regression
Random Forest is a versatile algorithm discussed in DATA 310 that can perform both regression and classification tasks. The regression tasks require picking a random set of points in the data and building Regression or **Decision Trees**, which are nested if-else conditions, and the splitting of a tree are decided based on criteria such as the gini impurity. 

In this example, we use a single Regression tree to predict the price of a car by using a combination of features such as wheelbase and horsepower.

*Example of a single Decision Tree*

![DecisionTreeExample](https://user-images.githubusercontent.com/98488236/153451668-f0f8905e-8bff-4673-a949-89316eb768ae.png)



*Python Regression Tree Example*

<img width="502" alt="RegressionTreePython" src="https://user-images.githubusercontent.com/98488236/153451519-b7b2f8c7-30d1-4987-9559-e8ae1750cd67.png">

Random Forest Regression also involves the concept of **bootstrapping**, which is sampling from a dataset with replacement and **feature bagging**, which is when a random subset of the feature dimensions is selected at each split in the growth of each Decision Trees. 

This means that at each split of the Decision tree, the model randomly considers only a small subset of features and the best split feature from the subset is used to split each node in a tree, unlike in regular **bagging** where all features are considered for the splitting of a node.

*Random Forest General Example*

![random forest](https://user-images.githubusercontent.com/98488236/153299496-9cbddba2-c965-4dd7-b8fa-d23941da6f47.png)
### Advantages of Random Forests
As previously stated, Random Forests are versatile and can perform both tasks in regression and classification. In the case of regression, the mean prediction of the trees is outputted for each forest grown, so unlike individual Decision Trees, Random Forests better avoids the problem of overfitting data.

Random Forest Regression also handles large datasets well and works especially well with non-linear data compared to other regression techniques.
### Disadvantages of Random Forests
Random Forests can be hard to visualize compared to single decision trees and often require large memory for storage.

Even though it is less likely to be overfit than individual Decision Trees, Random Forests can still overfit if the hyperparameters are not properly tuned, so we will try to optimize them in this project.

## Implementation of Random Forests
Sklearn has a Random Forest Regressor class that we import, and there are many hyperparemeters that we can tune to find a better model. For this project we will consider two hyperparameters for simplicity: number of trees in the forest and the maximum depth of each tree.


```
# import Random Forest Regressor from sklearn library
from sklearn.ensemble import RandomForestRegressor as RFR
rfr = RFR(n_estimators=150,max_depth=3)
```

We will later find optimal values for the number of trees and max depth

# Comparison of the Regression Techniques using 10-Fold validations
To find the cross-validated MSE of both regressions, two separate functions were created, since Lowess is not an SKlearn model.
```
# KFold function for Lowess Regression

def DoKFoldLoess(x,y,k, kern, tau, rseed):
  scale = SS()
  mse_lwr = []
  kf = KFold(n_splits=k,shuffle=True,random_state=rseed)
  for idxtrain, idxtest in kf.split(x):
    ytrain = y[idxtrain]
    xtrain = x[idxtrain]
    xstrain = scale.fit_transform(xtrain.reshape(-1,1))
    ytest = y[idxtest]
    xtest = x[idxtest]
    xstest = scale.transform(xtest.reshape(-1,1))
    yhat_lwr = lowess_reg(xstrain.ravel(),ytrain,xstest.ravel(),kern,tau)
    mse_lwr.append(mse(ytest,yhat_lwr))
  return np.mean(mse_lwr)

```

```
# KFold function for sklearn models

def DoKFold(model,x,y,k,rseed):
  scale = SS()
  mse_train = []
  mse_test = []
  kf = KFold(n_splits=k,shuffle=True,random_state=rseed)
  for idxtrain, idxtest in kf.split(x):
    ytrain = y[idxtrain]
    xtrain = x[idxtrain]
    xstrain = scale.fit_transform(xtrain.reshape(-1,1))
    ytest = y[idxtest]
    xtest = x[idxtest]
    xstest = scale.transform(xtest.reshape(-1,1))
    model.fit(xstrain,ytrain)
    mse_train.append(mse(ytrain,model.predict(xstrain)))
    mse_test.append(mse(ytest,model.predict(xstest)))
  return np.mean(mse_test)

```

# Optimizing the Hyperparameters
First, for Lowess Regression, to find the optimal kernel and value for tau, we perform multiple 10-Fold validations with different combinations of the two using three for-loops.
### Optimal Tau for Tricubic kernel
``` 
# find optimal tau for tricubic kernel

k = 10
t_range = np.arange(0.01,0.3, step=.01)
test_mse = []
for t in t_range:
  te_mse = DoKFoldLoess(x,y,k,tricubic, t, 410)
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
[0.2, 17.638049656558326]
![tricubicmse](https://user-images.githubusercontent.com/98488236/153296675-f4ae3c37-0607-4bfa-a19d-26145736ed74.png)

The same code was repeated, except switching the 'tricubic' kernel for the 'Epanechnikov' and 'quartic' kernels. The results and plots obtained were
### Optimal Tau for Epanechnikov kernel
[0.16, 17.640646421523392]
![epanechnikovmse](https://user-images.githubusercontent.com/98488236/153298160-b0e42109-eb4c-4a00-bc48-d92e8e1e2e50.png)


### Optimal Tau for Quartic kernel
[0.2, 17.645295248233868]
![quarticmse](https://user-images.githubusercontent.com/98488236/153298413-b28cf387-fbbf-4a0d-b827-4f38074cb55a.png)

Hence, we got that a tricubic kernel with a tau value of 0.2, produced the best cross-validated MSE of about 17.638.


## Optimizing Number of Trees and Max Depth for Random Forest
As for Random Forest, we found that a max_depth of 3 in conjunction with any value of n_estimators outperformed other values for max_depth and gave the lowest MSE values. Hence, all that was left was to find optimal values for n_estimators paired with a maximum depth of 3. This was done using a for loop and plotting the MSE values of the test sets to find the number of trees that produced the minimum value.

```
k = 10
t_range = np.arange(60,200, step=1)
test_mse = []
for t in t_range:
  rfr = RFR(n_estimators = t, max_depth = 3, random_state=410)
  te_mse = DoKFold(rfr,x,y,k,410)
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
[142, 17.866590990045857]

![MSE](https://user-images.githubusercontent.com/98488236/153284569-0e3435a5-5321-4866-a075-b9d390d09109.png)

Hence, we got that 142 trees in the forest with a maximum depth of 3 produced an MSE of about 17.866. 

Despite, trying our best to obtain a tau and the kernel that minimized the cross-validated test MSE, this value obtained for Random Forest Regression is still greater/worse than that obtained from Lowess Regression.
## Conclusion

By comparing the Locally Weighted Regression with Random Forest Regression on the "Cars" dataset, we found that, after tuning the hyperparameters for both Regression techniques, Lowess Regression produced a smaller cross-validated MSE: 17.638, than that produced by Random Forest Regression: 17.866. Moreover, even the suboptimal kernels implemented, Epanechnikov and quartic, also produced MSE values that were lower than Random Forest's 17.866.

Since we desire smaller MSE values, we can conclude that Locally Weighted Regression is superior to Random Forest Regression in this example using the cars dataset. Given the prevalence and importance of Random Forest, this project is a great introduction in demonstrating the significance and potential of Locally Weighted Regression.

## References
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3049417/

https://brilliant.org/wiki/multivariate-regression/#:~:text=Multivariate%20Regression%20is%20a%20method,responses)%2C%20are%20linearly%20related.&text=A%20mathematical%20model%2C%20based%20on,and%20other%20more%20complicated%20questions.

https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/

https://towardsdatascience.com/understanding-adaboost-2f94f22d5bfe

https://towardsdatascience.com/xgboost-python-example-42777d01001e

https://towardsdatascience.com/machine-learning-part-18-boosting-algorithms-gradient-boosting-in-python-ef5ae6965be4

https://www.geeksforgeeks.org/ml-gradient-boosting/
