# A Comparison Between Locally Weighted Regression and Random Forest Regression

## Locally Weighted Regression (Loess)
In locally weighted regression, points are weighted by proximity to the current x in question using a kernel, which in this case is a weighting function that the user decides on. Some examples of kernels include: Uniform, Tricubic, Epanechnikov (parabolic), and quartic (biweight). In this project we implemented the Tricubic, Epanechnikov, and quartic kernels in python as follows:
```
# tricubic kernel
def tricubic(x):
  if len(x.shape) == 1:  # use a conditional to reshape if it is a row vector
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,70/81*(1-d**3)**3)
```


```
def Epanechnikov(x):
  if len(x.shape) == 1:  
    x = x.reshape(-1,1)
  return np.where(np.abs(x)>1,0,3/4*(1-np.abs(x)**2)) 
```

```
def Quartic(x):
  if len(x.shape) == 1:  
    x = x.reshape(-1,1)
  return np.where(np.abs(x)>1,0,15/16*(1-np.abs(x)**2)**2) 
```
import matplotlib as mpl
import matplotlib.pyplot as plt


### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

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
