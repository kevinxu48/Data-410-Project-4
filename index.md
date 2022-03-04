# Comparison of Repeated Boosting Algorithms With LightGBM
## Multiple Boosting
In this project, we will update the algorithms created in Project 3, to implement repeated boosting. We will be repeatedly boosting Lowess Regression and compare the results with more boosting algorithms such as XGBoost and LightGBM.

## Implementation of Repeated Boosting

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
    'learning_rate': hp.quniform("learning_rate", 0.05, 0.4, 0.01),
    'max_depth': hp.choice("max_depth", np.arange(2, 10, 1, dtype=int)),
    'num_leaves': hp.choice("num_leaves", np.arange(4, 200, 4, dtype=int)),
    'reg_alpha': hp.uniform("reg_alpha", 2.0, 8.0),
    'reg_lambda': hp.uniform("reg_lambda", 2.0, 8.0),
    'subsample':        hp.uniform('subsample', 0.8, 1),
    'feature_fraction': hp.quniform("feature_fraction", 0.2, 0.8, 0.1),
    'n_estimators':     100,
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

({'feature_fraction': 0.6000000000000001,
  'learning_rate': 0.33,
  'max_depth': 7,
  'num_leaves': 39,
  'reg_alpha': 3.037722287779516,
  'reg_lambda': 7.3938913277804525,
  'subsample': 0.9154483095843352},
 <hyperopt.base.Trials at 0x7f8b6be51e10>)

## Cross Validation
To obtain cross-validated results, we will slightly modify the DoKFoldXGB function from Project 3 to perform cross validations for LightGBM. 

```
# KFold function for lgb

def DoKFoldLGB(X,y, obj, md, n_est, nl, rl, ra, lr,ff, s, random_state):
  mse_lgb = []
  scale = SS()
  kf = KFold(n_splits=10,shuffle=True,random_state=random_state)
  for idxtrain, idxtest in kf.split(X):
    xtrain = X[idxtrain]
    ytrain = y[idxtrain]
    ytest = y[idxtest]
    xtest = X[idxtest]
    xtrain = scale.fit_transform(xtrain)
    xtest = scale.transform(xtest)
    
    y_train = np.squeeze(ytrain)
    train_set = lgb.Dataset(xtrain, y_train, silent=True)
    
    model_lgb = lgb.LGBMRegressor(objective =obj,max_depth=md, n_estimators=n_est, learning_rate=lr, num_leaves = nl, reg_lambda=rl, reg_alpha=ra, 
                                     feature_fraction =ff, subsample = s,  random_state=410)
    #lgb.train()
    model_lgb.fit(xtrain,ytrain)
    yhat_lgb = model_lgb.predict(xtest)
    mse_lgb.append(mse(ytest,yhat_lgb))
  return np.mean(mse_lgb)
```

```
DoKFoldLGB(X,y,'regression',7,100,39,7.394,3.0377,0.33,0.6,.9154483095843352,410)
```

18.822660433997175

Thus, using Hyperopt to tune the hyperparameters for LightGBM Regression, we obtained a cross-validated MSE of 18.822660433997175, which is compared to our repeated boosting function.

## Conclusion

### References
https://www.kaggle.com/lasmith/house-price-regression-with-lightgbm

https://lightgbm.readthedocs.io/en/latest/Features.html

https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/

https://medium.com/optuna/lightgbm-tuner-new-optuna-integration-for-hyperparameter-optimization-8b7095e99258

https://towardsdatascience.com/feature-selection-in-machine-learning-using-lasso-regression-7809c7c2771a

https://towardsdatascience.com/an-example-of-hyperparameter-optimization-on-xgboost-lightgbm-and-catboost-using-hyperopt-12bc41a271e
