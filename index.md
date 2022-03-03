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
To tune the LightGBM hyperparameters, normally we would use a library such as Optuna, which contains various optimizers for hyperparameter tuning, but this requires a lot of coding and a long run time to obtain the optimal hyperparameters. Instead, we will use Hyperopt, which requires less coding and has a faster run time.

```
# import hyperopt and define lgb parameters
from hyperopt import hp
from sklearn.metrics import mean_squared_error as mse


lgb_reg_params = {
    'learning_rate':    hp.choice('learning_rate',    np.arange(0.05, 0.31, 0.05)),
    'max_depth':        hp.choice('max_depth',        np.arange(5, 16, 1, dtype=int)),
    'min_child_weight': hp.choice('min_child_weight', np.arange(1, 8, 1, dtype=int)),
    'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
    'min_data_in_leaf': hp.choice('min_data_in_leaf', np.arange(200, 10000, step=100)),
    'subsample':        hp.uniform('subsample', 0.8, 1),
    'num_leaves':       hp.choice('num_leaves', np.arange(20, 3000, step = 20)),
    'n_estimators':     1000,
    'bagging_fraction': hp.choice('bagging_fraction',   np.arange(0.2, 0.95, step=0.1)),
    'feature_fraction': hp.choice('feature_fraction',   np.arange(0.2, 0.95, step=0.1)),
}
lgb_fit_params = {
    'eval_metric': 'l2',
    'early_stopping_rounds': 10,
    'verbose': False
}
lgb_para = dict()
lgb_para['reg_params'] = lgb_reg_params
lgb_para['fit_params'] = lgb_fit_params
lgb_para['loss_func' ] = lambda y, pred: np.sqrt(mse(y, pred))
```

```
# instantiate the hyperopt class
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials


class HPOpt(object):

    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test  = x_test
        self.y_train = y_train
        self.y_test  = y_test

    def process(self, fn_name, space, trials, algo, max_evals):
        fn = getattr(self, fn_name)
        try:
            result = fmin(fn=fn, space=space, algo=algo, max_evals=max_evals, trials=trials)
        except Exception as e:
            return {'status': STATUS_FAIL,
                    'exception': str(e)}
        return result, trials

    def lgb_reg(self, para):
        reg = lgb.LGBMRegressor(**para['reg_params'])
        return self.train_reg(reg, para)

    def train_reg(self, reg, para):
        reg.fit(self.x_train, self.y_train,
                eval_set=[(self.x_train, self.y_train), (self.x_test, self.y_test)],
                **para['fit_params'])
        pred = reg.predict(self.x_test)
        loss = para['loss_func'](self.y_test, pred)
        return {'loss': loss, 'status': STATUS_OK}
```

```
# Find optimal hyperparameters on a single train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=410)

obj = HPOpt(X_train, X_test, y_train, y_test)
lgb_opt = obj.process(fn_name='lgb_reg', space=lgb_para, trials=Trials(), algo=tpe.suggest, max_evals=100)

```

*Optimal hyperparameters obtained from Hyperopt*

({'bagging_fraction': 5,
  'colsample_bytree': 3,
  'feature_fraction': 5,
  'learning_rate': 1,
  'max_depth': 7,
  'min_child_weight': 0,
  'min_data_in_leaf': 0,
  'num_leaves': 120,
  'subsample': 0.8504697538908891},
 <hyperopt.base.Trials at 0x7f2b2449a750>)
 

## Conclusion

### References
https://www.kaggle.com/lasmith/house-price-regression-with-lightgbm

https://lightgbm.readthedocs.io/en/latest/Features.html

https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/

https://medium.com/optuna/lightgbm-tuner-new-optuna-integration-for-hyperparameter-optimization-8b7095e99258

https://towardsdatascience.com/feature-selection-in-machine-learning-using-lasso-regression-7809c7c2771a

https://towardsdatascience.com/an-example-of-hyperparameter-optimization-on-xgboost-lightgbm-and-catboost-using-hyperopt-12bc41a271e
