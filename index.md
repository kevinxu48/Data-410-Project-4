# Comparison of Repeated Boosting Algorithms With LightGBM
## Data and Feature Selection
For this project we will again be using the concrete strength dataset, and to select the features. In project 3 we used dimensionality reduction, specifically PCA, to reduce the number of features. However, in this project we will be using Lasso (L1) Regularization for the purposes of feature selection. We can use the sklearn Lasso regularization and the Pipeline objects, which allows us to scale the data in advanced


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

## Multiple Boosting
In this project, we will update the algorithms created in Project 3, to implement repeated boosting. We will be repeatedly boosting Lowess Regression with Decision Trees and RandomForest regressors as the boosters. We will try different combinations of kernels and hyperparameters for the Random Forests, and then we will compare the results obtained from our repeated boosting algorithm with other boosting algorithms such as XGBoost and LightGBM.

## Implementation of Repeated Boosting
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
def boosted_lwr(X, y, xnew, kern, tau, intercept, model_boosting, nboost):
  # we need decision trees
  # for training the boosted method we use X and y
  Fx = lw_reg(X,y,X,kern,tau,intercept) # we need this for training the Decision Tree
  
  output = booster(X,y,xnew,kern,tau,model_boosting,nboost)
  return output 
```
```
#First boost on Lowess using RandomForest

mse_blwr = []

kf = KFold(n_splits=10,shuffle=True,random_state=410)
# this is the Cross-Validation Loop
for idxtrain, idxtest in kf.split(X):
  xtrain = X[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = X[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)
  dat_train = np.concatenate([xtrain,ytrain.reshape(-1,1)],axis=1)
  dat_test = np.concatenate([xtest,ytest.reshape(-1,1)],axis=1)
  #yhat_lwr = lw_reg(xtrain,ytrain, xtest,Epanechnikov,tau=0.9,intercept=True)
  #yhat_blwr = boosted_lwr(xtrain,ytrain, xtest,Epanechnikov,tau=0.9,intercept=True)
  yhat_blwr = boosted_lwr(xtrain,ytrain,xtest,Tricubic,1,True,model_boosting,2)
  #model_rf = RandomForestRegressor(n_estimators=100,max_depth=3)
    #model_rf.fit(xtrain,ytrain)
    #yhat_rf = model_rf.predict(xtest)
  model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=100,reg_lambda=20,alpha=1,gamma=10,max_depth=1)
  model_xgb.fit(xtrain,ytrain)
  yhat_xgb = model_xgb.predict(xtest)
  
  mse_blwr.append(mse(ytest,yhat_blwr))
    #mse_rf.append(mse(ytest,yhat_rf))
  mse_xgb.append(mse(ytest,yhat_xgb))
    ##mse_nn.append(mse(ytest,yhat_nn))
    #mse_NW.append(mse(ytest,yhat_sm))
#print('The Cross-validated Mean Squared Error for LWR is : '+str(np.mean(mse_lwr)))
print('The Cross-validated Mean Squared Error for Boosted LWR is : '+str(np.mean(mse_blwr)))
#print('The Cross-validated Mean Squared Error for RF is : '+str(np.mean(mse_rf)))
print('The Cross-validated Mean Squared Error for XGB is : '+str(np.mean(mse_xgb)))
#print('The Cross-validated Mean Squared Error for NN is : '+str(np.mean(mse_nn)))
#print('The Cross-validated Mean Squared Error for Nadarya-Watson Regressor is : '+str(np.mean(mse_NW)))
```



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
- Easy to access histograms for each leaf, which expedites the training procedure
- Lower memory usage than other boosters like XGBoost, since it allocates continuous values to discrete bins
### Disadvantages
- LightGBM splits the tree leaf-wise, which can lead to overfitting since it produces more complex trees than those produced by other boosting algorithms such as XGBoost. (this can be reduced by tuning the max_depth parameter)
- There are a lot of hyperparameters that require tuning in order to achieve better scores and higher accuracies 
## Hyperparameter Tuning
To tune the LightGBM hyperparameters, normally we would use a library such as Optuna, which contains various optimizers for hyperparameter tuning, but this requires a lot of coding and a long run time to obtain the optimal hyperparameters. Instead, we will use the Hyperopt library, which requires less coding and has a faster run time.

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

## XGBoost
## Hyperparameter tuning
We also used hyperopt to tune the hyperparameters for XGBoost to see how it compares to the repeated boosting algorithms and LightGBM. 

```
xgb_reg_params = {
    'max_depth': hp.choice('max_depth', np.arange(2, 10, 1, dtype=int)),
    'min_child_weight': hp.uniform('min_child_weight', 1, 8),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 0.8),
    'alpha': hp.quniform('alpha', 0.5, 1, 0.05),
    'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
    'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
    'n_estimators':     100,
    'eta': hp.quniform('eta', 0.025, 0.5, 0.01),
}
xgb_fit_params = {
    'eval_metric': 'rmse',
    'early_stopping_rounds': 10,
    'verbose': False
}
xgb_para = dict()
xgb_para['reg_params'] = xgb_reg_params
xgb_para['fit_params'] = xgb_fit_params
xgb_para['loss_func' ] = lambda y, pred: np.sqrt(mse(y, pred))

```
({'alpha': 0.8500000000000001,
  'colsample_bytree': 0.7371935662838315,
  'eta': 0.03,
  'gamma': 0.8500000000000001,
  'max_depth': 4,
  'min_child_weight': 3.3284732986473458,
  'subsample': 0.8500000000000001},
 <hyperopt.base.Trials at 0x7f96f1877e90>)

*Using the same KFold function for XGB as Project 3, we obtained these results:*

```
DoKFoldXGB(X,y, 'reg:squarederror',100,.85,0.85,4,0.03, 0.737, 0.85, 410, False)
```
22.157211602959503

This is noticably worse than the score obtained by LightGBM
## Conclusion
Even though we used Hyperopt to tune the hyperparameters for both LightGBM and XGBoost, LightGBM regression performed significantly better on the concrete strength dataset.
### References
https://www.kaggle.com/lasmith/house-price-regression-with-lightgbm

https://lightgbm.readthedocs.io/en/latest/Features.html

https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/

https://medium.com/optuna/lightgbm-tuner-new-optuna-integration-for-hyperparameter-optimization-8b7095e99258

https://towardsdatascience.com/feature-selection-in-machine-learning-using-lasso-regression-7809c7c2771a

https://towardsdatascience.com/an-example-of-hyperparameter-optimization-on-xgboost-lightgbm-and-catboost-using-hyperopt-12bc41a271e
