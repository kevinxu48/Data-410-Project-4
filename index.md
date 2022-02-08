# A Comparison Between Locally Weighted Regression and Random Forest Regression

## Locally Weighted Regression (Lowess)
In locally weighted regression, points are weighted by proximity to the current x in question using a kernel, which in this case is a weighting function that the user decides on. Some examples of kernels include: Uniform, Tricubic, Epanechnikov (parabolic), and quartic (biweight). In this project we implemented the Tricubic, Epanechnikov, and quartic kernels in python as follows:
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
def lowess_ag(x, y, f=2. / 3., iter=3):
    """lowess(x, y, f=2./3., iter=3) -> yest
    Lowess smoother: Robust locally weighted regression.
    The lowess function fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.
    The smoothing span is given by f. A larger value for f will result in a
    smoother curve. The number of robustifying iterations is given by iter. The
    function will run faster with a smaller number of iterations.
    """
    n = len(x)
    r = int(ceil(f * n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    yest = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(iter):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                          [np.sum(weights * x), np.sum(weights * x * x)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]

        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2

    return yest
```
```markdown
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
