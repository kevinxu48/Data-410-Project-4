# A Comparison Between Locally Weighted Regression and Random Forest Regression

# Locally Weighted Regression (Lowess)
In locally weighted regression, points are weighted by their proximity to a fitted data point using a kernel, which in this case is a weighting function that the user decides on, and will serve as one of two hyperparameters for our function. Some examples of kernels include: Uniform, Tricubic, Epanechnikov (parabolic), and quartic (biweight), but typically the algorithm uses a tri-cubic weight function. The kernel gives the most weight to the data points nearest to the point of estimation and the least weight to the data points that are furthest away. 

The other hyperparameter in our function is tau, which is called the "bandwidth" or the "smoothing parameter" and it controls how flexible the Lowess function will be by determining how much of the data will be used to fit each local curve.

### Implementation of Kernels
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
Random Forest is a Regression technique discussed in DATA 310, which involves picking a random set of points in the data and building regression trees. In the case of regression, the mean prediction of the trees is outputted for each forest grown. Sklearn has a Random Forest Regressor class that we import, and there are two hyperparameters that we will consider: number of trees in the forest and the maximum depth of each tree.

![random forest](https://user-images.githubusercontent.com/98488236/153299496-9cbddba2-c965-4dd7-b8fa-d23941da6f47.png)



```
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=150,max_depth=3)
```
Syntax highlighted code block

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
To find the cross-validated MSE of both regressions
# Optimizing the Hyperparameters
First, for Lowess Regression, to find the optimal kernel and value for tau, we perform multiple 10-Fold validations with different combinations of the two using three for-loops.
## Optimal Tau for Tricubic kernel
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
First, we found that a max_depth of 3 in conjunction with , gave the lowest MSE values for Random Forest, so all that was left was to find optimal values for n_estimators. This was done using a for loop and plotting 

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

Hence, we got that 142 trees in the forest with a maximum depth of 3 produced an MSE of about 17.866. Despite, trying our best to obtain hyperparemters that minimized the cross-validated test MSE, this value is still lower than that obtained from Lowess Regression.
## Conclusion

By comparing the Locally Weighted Regression with Random Forest Regression on the "Cars" dataset, we found that The Loess produced a smaller cross-validated MSE than that of Random Forest. Since we desire smaller MSE values, we can conclude that Locally Weighted Regression is superior to Random Forest Regression in this example. Given the prevalence and importance of Random Forest, this project demonstrates the significance and potential of Locally Weighted Regression.
