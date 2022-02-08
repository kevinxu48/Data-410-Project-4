# A Comparison Between Locally Weighted Regression and Random Forest Regression

## Locally Weighted Regression (Lowess)
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
## Random Forest Regression
Random Forest is a Regression technique discussed in DATA 310, which involves picking a random set of points in the data and building regression trees. In the case of regression, the mean prediction of the trees is outputted for each forest grown. Sklearn has a Random Forest Regressor class that we import, and there are three hyperparameters that we will consider: number of trees in the forest, the maximum depth of the tree, and the minimum number of samples required to split an internal node.

To find the optimal values for the hyperparameters, we

```
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=150,max_depth=3)
```
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Conclusion

By comparing the Locally Weighted Regression with Random Forest Regression on the "Cars" dataset, we found that The Loess produced a smaller cross-validated MSE than that of Random Forest. Since we desire smaller MSE values, we can conclude that Locally Weighted Regression is superior to Random Forest Regression in this example. Given the prevalence and importance of Random Forest, this project demonstrates the significance and potential of Locally Weighted Regression.
### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
