# A Comparison Between Multivariate Regression Analysis and Gradient Boosting

## Multivariate Regression Analysis
Multivariate regression is a regression technique that estimates a single regression model with more than one outcome variable. When there is more than one predictor variable in a multivariate regression model, the model is a multivariate multiple regression.
*Example of how the Locally Weighted Regression works*

![lowess-pred-1](https://user-images.githubusercontent.com/98488236/153333566-4816a54d-b7c3-47ad-87be-aabefcccf3c6.gif)


Some examples of kernels include: Uniform, Tricubic, Epanechnikov (parabolic), and quartic (biweight), but typically the algorithm uses a tri-cubic weight function. The kernel gives the most weight to the data points nearest to the point of estimation and the least weight to the data points that are furthest away. 

The other hyperparameter in our function is tau, which is called the "bandwidth" or the "smoothing parameter" and it controls how flexible the Lowess function will be by determining how much of the data will be used to fit each local curve.

*Example of Loess with tricubic kernel and y as a sine function*
![lowess](https://user-images.githubusercontent.com/98488236/153338162-96c1cccc-4086-46b9-9e77-3903639acf5a.png)
We can see that Lowess Regression with a tricubic kernel was able to almost perfectly fit a model to the sine function y, despite the noise, making it a strong learner model.
### Advantages of Lowess
The main advantage and appeal of Lowess is the fact that it does not require the specification of a global function to fit a model to the entire dataset, only subsets of the data. Instead we only have to choose a kernel and a tau value.

### Disadvantages of Lowess
A big limitation of Lowess is that you need a relatively large sampled dataset to produce useful models because it needs to observe a sufficient amount of data subsets to perform local fitting, which can be imprecise if there is not enough data.

In addition, since Lowess is a Least Squares method, it is also hurt by the presence of outliers, since they would still have a great effect on the slope of the locally fitted regression line. 

Lowess is also a computationally intensive method, since a regression model is computed for each point.
## Implementation of Kernels
In this project we defined the Tricubic, Epanechnikov, and quartic kernels in python as follows:
```
# Tricubic kernel
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
