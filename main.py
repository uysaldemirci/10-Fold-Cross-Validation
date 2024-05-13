import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.layers import Input
from keras.optimizers import Adam

# Load the dataset
dataset = pd.read_csv('Gunsnew.csv')
#dataset.info()

# Shows first 5 rows of dataset
#print(dataset.head())

# Split dependent and independent variables
X = dataset.iloc[:, 0:12]  # Independent variables
y = dataset.iloc[:, 12]    # Dependent variable

# Convert states to continuous variables
state_mapping = {state: idx + 1 for idx, state in enumerate(X['state'].unique())}
X['state'] = X['state'].map(state_mapping)

# Convert law categorical data to continuous variables
y = y.map({'yes': 1, 'no': 0})


# Split the X and Y Dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling for normalization
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

# Function to create ANN model
def create_model():
    model = Sequential()
    model.add(Input(shape=(12,)))
    model.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    optimizer = Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create KerasClassifier
classifier = KerasClassifier(model=create_model, batch_size=10, epochs=100, verbose=0)

# Define the pipeline
pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', classifier)])



# Perform 10-fold cross validation with detailed results
from sklearn.model_selection import cross_validate

# Perform 10-fold cross validation
cv_results = cross_validate(estimator=pipeline, X=X_train, y=y_train, cv=10, scoring=['accuracy', 'neg_log_loss'])

# Extracting results
accuracy_scores = cv_results['test_accuracy']
cost_function_values = -cv_results['test_neg_log_loss']  # Negated as the function returns negative log loss

# Print accuracy scores at each fold
print("Accuracy Scores at Each Fold:")
for fold, accuracy in enumerate(accuracy_scores, start=1):
    print(f"Fold {fold}: {accuracy}")

# Print cost function values at each fold
print("\nCost Function Values at Each Fold:")
for fold, cost in enumerate(cost_function_values, start=1):
    print(f"Fold {fold}: {cost}")

# Print averaged cost function value
average_cost = cost_function_values.mean()
print(f"\nAverage Cost Function Value: {average_cost}")

# Print average accuracy
print("Average Accuracy: {:.2f}%".format(accuracy_scores.mean() * 100))