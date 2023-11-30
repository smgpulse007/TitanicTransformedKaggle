import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC


# Load data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
# Example of creating interaction features

# Total expenditure
train_data['TotalExpenditure'] = train_data[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
test_data['TotalExpenditure'] = test_data[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)

# CryoSleep and VIP interaction
train_data['CryoSleep_and_VIP'] = train_data['CryoSleep'] & train_data['VIP']
test_data['CryoSleep_and_VIP'] = test_data['CryoSleep'] & test_data['VIP']


# Bin Age and TotalExpenditure
train_data['AgeBin'] = pd.qcut(train_data['Age'], 4, labels=False)
test_data['AgeBin'] = pd.qcut(test_data['Age'], 4, labels=False, duplicates='drop')

train_data['ExpenditureBin'] = pd.qcut(train_data['TotalExpenditure'], 4, labels=False, duplicates='drop')
test_data['ExpenditureBin'] = pd.qcut(test_data['TotalExpenditure'], 4, labels=False, duplicates='drop')


# Separate target from predictors
y = train_data["Transported"]
X = train_data.drop(["Transported"], axis=1)

# Select categorical columns with relatively low cardinality
categorical_cols = [cname for cname in X.columns if
                    X[cname].nunique() < 10 and 
                    X[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X.columns if 
                  X[cname].dtype in ['int64', 'float64']]

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='mean')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define base models
model1 = RandomForestClassifier(random_state=0)


# Pipelines for each model
pipeline1 = Pipeline(steps=[('preprocessor', preprocessor), ('model', model1)])

# Define the parameter grid
param_grid = {
    'model__n_estimators': [87,113,269],
    'model__max_depth': [7,9,11],
    'model__min_samples_split': [2,3,5],
    'model__min_samples_leaf': [6,7,11],
    'model__max_features': ['sqrt']
}

# Perform grid search
gs = GridSearchCV(pipeline1, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
gs.fit(X, y)

# Print the best parameters and score
print("Best parameters:", gs.best_params_)
print("Best score:", gs.best_score_)

# Extract best parameters for RandomForestClassifier
best_params = gs.best_params_
best_rf = RandomForestClassifier(
    n_estimators=best_params['model__n_estimators'],
    max_depth=best_params['model__max_depth'],
    min_samples_split=best_params['model__min_samples_split'],
    min_samples_leaf=best_params['model__min_samples_leaf'],
    max_features=best_params['model__max_features'],
    random_state=0
)
# Define other base models
model2 = LogisticRegression(max_iter=1000, random_state=0)
model3 = GradientBoostingClassifier(random_state=0)

# Pipelines for each model
pipeline1 = Pipeline(steps=[('preprocessor', preprocessor), ('model', best_rf)])
pipeline2 = Pipeline(steps=[('preprocessor', preprocessor), ('model', model2)])
pipeline3 = Pipeline(steps=[('preprocessor', preprocessor), ('model', model3)])

# Define the Voting Classifier
voting_clf = VotingClassifier(estimators=[
    ('rf', pipeline1), 
    ('lr', pipeline2), 
    ('gb', pipeline3)
], voting='soft')

# Fit the Voting Classifier
voting_clf.fit(X, y)

# Predict on test data
predictions = voting_clf.predict(test_data)

from sklearn.model_selection import cross_val_score, StratifiedKFold

# Stratified K-Fold cross-validation
strat_k_fold = StratifiedKFold(n_splits=5)

# Evaluate the model
scores = cross_val_score(voting_clf, X, y, cv=strat_k_fold)
print("CV Scores:", scores)

# Save test predictions to file
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Transported': predictions})
output.to_csv('submission.csv', index=False)

