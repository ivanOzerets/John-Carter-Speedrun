```python
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')

RANDOM_STATE = 55 ## We will pass it to every sklearn call so we ensure reproducibility
```

Introduction

- dataset
    - this dataset is obtained from Kaggle: [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- context
    - cardiovascular disease (CVDs) is the number one cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide
    - four out of five CVD deaths are due to heart attacks and strokes, and one-third of these deaths occur prematurely in people under 70 years of age
    - heart failure is a common event caused by CVDs
    - people with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management
    - this dataset contains 11 features that can be used to predict possible heart disease
    - train a machine learning model to assist with diagnosing this disease
- attribute information
    - Age: age of the patient [years]
    - Sex: sex of the patient [M: Male, F: Female]
    - ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
    - RestingBP: resting blood pressure [mm Hg]
    - Cholesterol: serum cholesterol [mm/dl]
    - FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
    - RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
    - MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
    - ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
    - Oldpeak: oldpeak = ST [Numeric value measured in depression]
    - ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
    - HeartDisease: output class [1: heart disease, 0: Normal]
- load the dataset with variables
    - Sex
    - ChestPainType
    - RestingECG
    - ExerciseAngina
    - ST_Slope

    ```python
    # Load the dataset using pandas
    df = pd.read_csv("heart.csv")

    df.head()
    ```

- perform some data engineering before working with models
- there are five categorical features, so use Pandas to one-hot encode them

One-hot encoding using Pandas

- remove the binary variables, because one-hot encoding them would do nothing to them
- to achieve this, just count how many different values there are in each categorical variable and consider only the variables with 3 or more values

    ```python
    cat_variables = ['Sex',
    'ChestPainType',
    'RestingECG',
    'ExerciseAngina',
    'ST_Slope'
    ]
    ```

- one-hot encoding aims to transform a categorical variable with n outputs into n binary variables
- pandas has a built-in method to one-hot encode variables, it is the function pd.get_dummies
- there are several arguments to this function, here are a few
    - data: DataFrame to be used
    - prefix: a list with prefixes, so you know which value we are dealing with
    - columns: the list of columns that will be one-hot encoded, ‘prefix’ and ‘columns’ must have the same length
- for more information, type help(pd.get_dummies) to read the function’s full docs

    ```python
    # This will replace the columns with the one-hot encoded ones and keep the 
    # columns outside 'columns' argument as it is.
    df = pd.get_dummies(data = df,
                             prefix = cat_variables,
                             columns = cat_variables)

    df.head()
    ```

- choose the variables that will be the input features of the model
    - the target is HeartDisease
    - all other variables are featurs that can potentially be used to predict the target

    ```python
    features = [x for x in df.columns if x not in 'HeartDisease'] 
    ## Removing our target variable
    ```

- started with 11 features, see how many features variables there are after one-hot encoding

    ```python
    print(len(features))

    # 20
    ```

Splitting the Dataset

- split the dataset into train and test datasets
- use the function train_test_split from Scikit-learn

```python
X_train, X_val, y_train, y_val = train_test_split(df[features], df['HeartDisease'], train_size = 0.8, random_state = RANDOM_STATE)

# We will keep the shuffle = True since our dataset has not any time dependency.
```

```python
print(f'train samples: {len(X_train)}')
print(f'validation samples: {len(X_val)}')
print(f'target proportion: {sum(y_train)/len(y_train):.4f}')

'''
train samples: 734
validation samples: 184
target proportion: 0.5518
'''
```

Building the models

- Decision Tree
    - there are several hyperparameters in the decision tree object from scikit-learn
    - only be using some of them and also will not perform feature selection nor hyperparameter tuning
    - the hyperparameters that will used and investigated
        - min_samples_split: the minimum number of samples required to split an internal node
            - choosing a higher min_samples_split can reduce the number of splits and may help to reduce overfitting
            - max_depth: the maximum depth of the tree
                - choosing a lower max_depth can reduce the number of splits and may help to reduce overfitting

        ```python
        min_samples_split_list = [2,10, 30, 50, 100, 200, 300, 700] ## If the number is an integer, then it is the actual quantity of samples,
        max_depth_list = [1,2, 3, 4, 8, 16, 32, 64, None] # None means that there is no depth limit.
        ```

        ```python
        accuracy_list_train = []
        accuracy_list_val = []
        for min_samples_split in min_samples_split_list:
            # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
            model = DecisionTreeClassifier(min_samples_split = min_samples_split,
                                           random_state = RANDOM_STATE).fit(X_train,y_train) 
            predictions_train = model.predict(X_train) ## The predicted values for the train dataset
            predictions_val = model.predict(X_val) ## The predicted values for the test dataset
            accuracy_train = accuracy_score(predictions_train,y_train)
            accuracy_val = accuracy_score(predictions_val,y_val)
            accuracy_list_train.append(accuracy_train)
            accuracy_list_val.append(accuracy_val)

        plt.title('Train x Validation metrics')
        plt.xlabel('min_samples_split')
        plt.ylabel('accuracy')
        plt.xticks(ticks = range(len(min_samples_split_list )),labels=min_samples_split_list)
        plt.plot(accuracy_list_train)
        plt.plot(accuracy_list_val)
        plt.legend(['Train','Validation'])
        ```

    - note how increasing the number of min_samples_split reduces overfitting
        - increasing min_samples_split from 10 to 30, and from 30 to 50, even though it doesn’t improve the validation accuracy, it brings the training accuracy closer to it, showing a reduction in overfitting
    - again with max_depth

        ```python
        accuracy_list_train = []
        accuracy_list_val = []
        for max_depth in max_depth_list:
            # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
            model = DecisionTreeClassifier(max_depth = max_depth,
                                           random_state = RANDOM_STATE).fit(X_train,y_train) 
            predictions_train = model.predict(X_train) ## The predicted values for the train dataset
            predictions_val = model.predict(X_val) ## The predicted values for the test dataset
            accuracy_train = accuracy_score(predictions_train,y_train)
            accuracy_val = accuracy_score(predictions_val,y_val)
            accuracy_list_train.append(accuracy_train)
            accuracy_list_val.append(accuracy_val)

        plt.title('Train x Validation metrics')
        plt.xlabel('max_depth')
        plt.ylabel('accuracy')
        plt.xticks(ticks = range(len(max_depth_list )),labels=max_depth_list)
        plt.plot(accuracy_list_train)
        plt.plot(accuracy_list_val)
        plt.legend(['Train','Validation'])
        ```

    - in general, reducting max_depth can help to reduce overfitting
        - reducing max_depth from 8 to 4 increases validation accuracy closer to training accuracy, while significantly reducing training accuracy
        - the validation accuracy reaches the highest at tree_depth=4
        - when the max_depth is smaller than 3, both training and validation accuracy decreases
        - the tree cannot make enough splits to distinguish positive from negatives (the model is underfitting the training set)
        - when the max_depth is too high (≥ 5), validation accuracy decreases while training accuracy increases, indicating that the model is overfitting to the training set
    - choose the best values for these two hyper-parameters
        - max_depth = 4
        - mi_samples_split = 50

        ```python
        decision_tree_model = DecisionTreeClassifier(min_samples_split = 50,
                                                     max_depth = 3,
                                                     random_state = RANDOM_STATE).fit(X_train,y_train)
        ```

        ```python
        print(f"Metrics train:\n\tAccuracy score: {accuracy_score(decision_tree_model.predict(X_train),y_train):.4f}")
        print(f"Metrics validation:\n\tAccuracy score: {accuracy_score(decision_tree_model.predict(X_val),y_val):.4f}")

        '''
        Metrics train:
        	Accuracy score: 0.8583
        Metrics validation:
        	Accuracy score: 0.8641
        '''
        ```

    - metrics are not good, by no overfitting

Random Forest

- try random forest alg
    - all of the hyperparameters found in the decision tree model will also exist in this algorithm, since a random forest is an ensemble of many Decision Trees
    - one additional hyperparameter for Random Forest is called n_estimators which is the number of Decision Trees that make up the Random Forest
- for a random forest, we randomly choose a subset of the features and randomly choose a subset of the training examples to train each individual tree
    - following the lectures, if n is the number of features, we will randomly select sqrt(n) of these features to train each individual tree
    - note that you can modify this by setting the max_features parameter.
- can also speed up the training jobs with another parameters, n_jobs
    - since the fitting of each tree is independent of each other, it is possible fit more than one tree in parallel
    - so setting n_jobs higher will increase how many CPU cores it will use
    - note that the numbers very close to the maximum cores of your CPU may impact on the overall performance of your PC and even lead to freezes
    - changing this parameter does not impact on the final result but can reduce the training time
- run the same script again, but with another parameter, n_estimators, chooseing between 10, 50, and 100, 100 is default

    ```python
    min_samples_split_list = [2,10, 30, 50, 100, 200, 300, 700]  ## If the number is an integer, then it is the actual quantity of samples,
                                                 ## If it is a float, then it is the percentage of the dataset
    max_depth_list = [2, 4, 8, 16, 32, 64, None]
    n_estimators_list = [10,50,100,500]
    ```

    ```python
    accuracy_list_train = []
    accuracy_list_val = []
    for min_samples_split in min_samples_split_list:
        # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
        model = RandomForestClassifier(min_samples_split = min_samples_split,
                                       random_state = RANDOM_STATE).fit(X_train,y_train) 
        predictions_train = model.predict(X_train) ## The predicted values for the train dataset
        predictions_val = model.predict(X_val) ## The predicted values for the test dataset
        accuracy_train = accuracy_score(predictions_train,y_train)
        accuracy_val = accuracy_score(predictions_val,y_val)
        accuracy_list_train.append(accuracy_train)
        accuracy_list_val.append(accuracy_val)

    plt.title('Train x Validation metrics')
    plt.xlabel('min_samples_split')
    plt.ylabel('accuracy')
    plt.xticks(ticks = range(len(min_samples_split_list )),labels=min_samples_split_list) 
    plt.plot(accuracy_list_train)
    plt.plot(accuracy_list_val)
    plt.legend(['Train','Validation'])
    ```

- notice that, even though the validation accuracy reached is the same both at min_sample_split = 2 and 10, in the latter, difference in training and validation set reduces, showing less overfitting

    ```python
    accuracy_list_train = []
    accuracy_list_val = []
    for max_depth in max_depth_list:
        # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
        model = RandomForestClassifier(max_depth = max_depth,
                                       random_state = RANDOM_STATE).fit(X_train,y_train) 
        predictions_train = model.predict(X_train) ## The predicted values for the train dataset
        predictions_val = model.predict(X_val) ## The predicted values for the test dataset
        accuracy_train = accuracy_score(predictions_train,y_train)
        accuracy_val = accuracy_score(predictions_val,y_val)
        accuracy_list_train.append(accuracy_train)
        accuracy_list_val.append(accuracy_val)

    plt.title('Train x Validation metrics')
    plt.xlabel('max_depth')
    plt.ylabel('accuracy')
    plt.xticks(ticks = range(len(max_depth_list )),labels=max_depth_list)
    plt.plot(accuracy_list_train)
    plt.plot(accuracy_list_val)
    plt.legend(['Train','Validation'])
    ```

    ```python
    accuracy_list_train = []
    accuracy_list_val = []
    for n_estimators in n_estimators_list:
        # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
        model = RandomForestClassifier(n_estimators = n_estimators,
                                       random_state = RANDOM_STATE).fit(X_train,y_train) 
        predictions_train = model.predict(X_train) ## The predicted values for the train dataset
        predictions_val = model.predict(X_val) ## The predicted values for the test dataset
        accuracy_train = accuracy_score(predictions_train,y_train)
        accuracy_val = accuracy_score(predictions_val,y_val)
        accuracy_list_train.append(accuracy_train)
        accuracy_list_val.append(accuracy_val)

    plt.title('Train x Validation metrics')
    plt.xlabel('n_estimators')
    plt.ylabel('accuracy')
    plt.xticks(ticks = range(len(n_estimators_list )),labels=n_estimators_list)
    plt.plot(accuracy_list_train)
    plt.plot(accuracy_list_val)
    plt.legend(['Train','Validation'])
    ```

- fit a random forest with the best parameters
    - max_depth: 16
    - min_samples_split: 10
    - n_estimators: 100

    ```python
    random_forest_model = RandomForestClassifier(n_estimators = 100,
                                                 max_depth = 16, 
                                                 min_samples_split = 10).fit(X_train,y_train)
    ```

    ```python
    print(f"Metrics train:\n\tAccuracy score: {accuracy_score(random_forest_model.predict(X_train),y_train):.4f}\nMetrics test:\n\tAccuracy score: {accuracy_score(random_forest_model.predict(X_val),y_val):.4f}")

    '''
    Metrics train:
    	Accuracy score: 0.9319
    Metrics test:
    	Accuracy score: 0.8859
    '''
    ```

- searching for the best value one hyperparameter while leaving the other hyperparameters at their default values
    - ideally, we would want to check every combination of values for every hyperparameter that we are tuning
    - if we have 3 hyperparameters, and each hyperparameter has 4 values to try out, we should have a total of 4 x 4 x 4 = 64 combinations to try
    - when we only modify one hyperparameter while leaving the rest as their default value, we are trying 4 + 4 + 4 = 12 results
    - to try out all combinations, we can use a sklearn implementation called GridSearchCV
    - GridSearchCV has a refit parameter that will automatically refit a model on the best combination so we will not need to program it explicitly, please refer to its [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

XGBoost

- gradient boosting model
- boosting methods train several trees, but instead of them being uncorrelated to each other, now the trees are in one after the other in order to minimize the error
- the model has the same parameters as a decision tree, plus the learning rate
    - the learning rate is the size of the step on the GD method that the XGBoost uses internally to minimize the error on each train step
- one interesting thing about the XGBoost is that during fitting, it can take in an evaluation dataset of the form (X_val, y_val)
    - on each iteration, it measures the cost (or evaluation metric) on the evaluation datasets
    - once the cost (or metric) stops decreasing for a number of rounds (called early_stopping_rounds), the training will stop
    - more iterations lead to more estimators, and more estimators can result in overfitting
    - by stopping once the validation metric no longer improves, you can limit the number of estimators created, and reduce overfitting
- define a subset of the training set

    ```python
    n = int(len(X_train)*0.8) ## Let's use 80% to train and 20% to eval

    X_train_fit, X_train_eval, y_train_fit, y_train_eval = X_train[:n], X_train[n:], y_train[:n], y_train[n:]
    ```

- set a large number of estimators
- stop if the cost function stops decreasing
- note some of the .fit() parameters
    - eval_set = [(X_train_eval,y_train_eval)] → pass a list to the eval_set, because you can have several different tuples of eval sets
- early_stopping_rounds → helps to stop the  model training if its evaluation metric is no longer improving on the validation set, set to 10
    - the model keeps track of the round with the best performance (lowest evaluation metric)
    - for example, say round 16 has the lowest evaluation metric so far
    - each successive round’s evaluation metric is compared to the best metric
    - if the model goes 10 rounds where none have a better metric than the best one, then the model stops training
    - the model is returned at its last state when training terminated, not its state during the best round
    - for example, if the model stops at round 26, but the best round was 16, the model’s training state at round 26 is returned, not round 16
    - note that this is different from returning the model’s ‘best’ state (from when the evaluation metric was the lowest)

    ```python
    xgb_model = XGBClassifier(n_estimators = 500, learning_rate = 0.1,verbosity = 1, random_state = RANDOM_STATE)
    xgb_model.fit(X_train_fit,y_train_fit, eval_set = [(X_train_eval,y_train_eval)], early_stopping_rounds = 10)

    '''
    [0]	validation_0-logloss:0.64479
    [1]	validation_0-logloss:0.60569
    ...
    [25]	validation_0-logloss:0.45383
    [26]	validation_0-logloss:0.45547
    XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
                  colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
                  early_stopping_rounds=None, enable_categorical=False,
                  eval_metric=None, gamma=0, gpu_id=-1, grow_policy='depthwise',
                  importance_type=None, interaction_constraints='',
                  learning_rate=0.1, max_bin=256, max_cat_to_onehot=4,
                  max_delta_step=0, max_depth=6, max_leaves=0, min_child_weight=1,
                  missing=nan, monotone_constraints='()', n_estimators=500,
                  n_jobs=0, num_parallel_tree=1, predictor='auto', random_state=55,
                  reg_alpha=0, reg_lambda=1, ...)
    '''
    ```

- even though the model was initialized to allow up to 500 estimators, the alg only fit 26 estimators (over 26 rounds of training)
- to see why, look for the round of training that had the best performance (lowest evaluation metric
- view the validation log loss metrics that were output above or view the model’s .best_iteration attribute:

    ```python
    xgb_model.best_iteration

    # 16
    ```

- the best round of training was round 16, with a log loss of 4.3948
    - for 10 rounds of training after that, the log loss was higher than this
    - since early_stopping_rounds was set to 10, by the 10th round where the log loss doesn’t improve upon the best one, training stops
    - try out different values of early_stopping_rounds to verify
    - if you set to 20, the model stops training at round 36

    ```python
    print(f"Metrics train:\n\tAccuracy score: {accuracy_score(xgb_model.predict(X_train),y_train):.4f}\nMetrics test:\n\tAccuracy score: {accuracy_score(xgb_model.predict(X_val),y_val):.4f}")

    '''
    Metrics train:
    	Accuracy score: 0.9251
    Metrics test:
    	Accuracy score: 0.8641
    '''
    ```