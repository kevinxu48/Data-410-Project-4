# Comparison of Repeated Boosting Algorithms With LightGBM
## Multiple Boosting
In this project, we will update the algorithms created in Project 3, to implement repeated boosting. Compare with more boosting algorithms such as LightGBM

## Data and Feature Selection
For this project we will again be using the concrete strength dataset, and to select the features. However in project 3 we used dimensionality reduction, specifically PCA, to reduce the number of features. In this project, we will be using Lasso (L1) Regularization for the purposes of feature selection. We can use the sklearn Lasso regularization and the Pipeline objects, which allows us to scale the data in advanced


```
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso

# use Pipeline to scale the data before performing L1 Regularization
pipeline = Pipeline([
                     ('scaler',SS()),
                     ('model',Lasso())
])
```

```
# Perform a single train-test split before doing regularization

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=410)

search.fit(X_train,y_train)

coefficients = search.best_estimator_.named_steps['model'].coef_
importance = np.abs(coefficients)
importance
```

```
# see what features are kept
print("The features kept are: " + str(concrete.drop(columns = ['strength']).columns[importance > 0]))
print("The features removed are: " + str(concrete.drop(columns = ['strength']).columns[importance == 0]))
```

The features kept are: Index(['cement', 'slag', 'ash', 'water', 'superplastic', 'age'], dtype='object')

The features removed are: Index(['coarseagg', 'fineagg'], dtype='object')

Thus, we will proceed with 6 out of the 8 features: cement, slag, ash, water, superplastic, and age.

# Light Gradient Boosting Machine (LightGBM) Regressor
LightGBM is an open source gradient boosting algorithm proposed and developed by Microsoft. Compared to XGBoost which grows Decision Trees based on a pre-sort-based algorithms, LightGBM instead uses histogram-based algorithms. LightGBM provides the following distributed learning algorithms: Feature parallel and Data parallel.


In this project we will implement the LightGBM regressor. The LightGBM algorithm utilizes two techniques called Gradient-Based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB) which allow the algorithm to run faster while maintaining a high level of accuracy

The python implementation of LightGBM can use a dictionary to tune the hyperparameters.

*Illustration of LightGBM Decision Tree Growth*

![lightgbm tree growth](https://user-images.githubusercontent.com/98488236/156248524-17d3be55-97f7-43ed-9691-e5552ba1c19d.png)

*Illustration of XGBoost Decision Tree Growth*

![xgboost tree growth](https://user-images.githubusercontent.com/98488236/156248659-ab655193-601b-4df8-8c84-b5e6854d9bad.png)

### Advantages
 - Reduced cost of calculating the gain for each split
 - Easy to access histograms for each leaf
### Disadvantages

## Hyperparameter Tuning
To tune the LightGBM hyperparameters, we will use the Optuna library, which contains various optimizers for hyperparameter tuning.

## Conclusion

### References
https://www.kaggle.com/lasmith/house-price-regression-with-lightgbm

https://lightgbm.readthedocs.io/en/latest/Features.html

https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/

https://optunity.readthedocs.io/en/latest/

https://medium.com/optuna/lightgbm-tuner-new-optuna-integration-for-hyperparameter-optimization-8b7095e99258

https://towardsdatascience.com/feature-selection-in-machine-learning-using-lasso-regression-7809c7c2771a
