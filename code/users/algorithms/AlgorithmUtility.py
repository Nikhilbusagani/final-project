import pandas as pd
from sklearn.model_selection import train_test_split
from django.conf import settings
from sklearn.metrics import precision_score,mean_absolute_error,mean_squared_error,r2_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from tensorflow import keras
from tensorflow.keras import layers                                                         

path = settings.MEDIA_ROOT + "//" + "ElectricCarData_Modified.csv"
df = pd.read_csv(path)
print(df.columns)
y = df['PriceEuro']  # Assuming 'PriceEuro' is the target
X = df.drop(columns=['PriceEuro'])
num_features = ['AccelSec', 'TopSpeed_KmH', 'Range_Km', 'Battery_Pack Kwh',
                'Efficiency_WhKm', 'FastCharge_KmH', 'Seats']
cat_features = ['RapidCharge', 'PowerTrain', 'PlugType', 'BodyStyle', 'Segment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Standardize numerical features
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[num_features])
X_test_num = scaler.transform(X_test[num_features])

# One-hot encode categorical features
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
X_train_cat = encoder.fit_transform(X_train[cat_features])
X_test_cat = encoder.transform(X_test[cat_features])

# Combine processed numerical and categorical data
X_train = np.hstack((X_train_num, X_train_cat))
X_test = np.hstack((X_test_num, X_test_cat))

def calc_linear_regression():
    print("*" * 25, "linear Regression Classification")
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)  # Trained wih 80% Data
    y_pred = model.predict(X_test)
    r21 = r2_score(y_test, y_pred)
    print('lg r2_score:', r21)
    mae = mean_absolute_error(y_test, y_pred)
    print('lg mae:', mae)
    mse = mean_squared_error(y_test, y_pred)
    print('LG mse:', mse)
   
    return r21,mae,mse


def calc_decision_tree():
    print("*" * 25, "Decision Tree Classification")
    from sklearn.tree import DecisionTreeRegressor
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)  # Trained wih 80% Data
    y_pred = model.predict(X_test)
    r21 = r2_score(y_test, y_pred)
    print('lg r2_score:', r21)
    mae = mean_absolute_error(y_test, y_pred)
    print('lg mae:', mae)
    mse = mean_squared_error(y_test, y_pred)
    print('LG mse:', mse)
   
    return r21,mae,mse


def calc_random_forest():
    print("*" * 25, "Random Forest Classification")
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor()
    model.fit(X_train, y_train)  # Trained wih 80% Data
    y_pred = model.predict(X_test)
    r21 = r2_score(y_test, y_pred)
    print('lg r2_score:', r21)
    mae = mean_absolute_error(y_test, y_pred)
    print('lg mae:', mae)
    mse = mean_squared_error(y_test, y_pred)
    print('LG mse:', mse)
   
    return r21,mae,mse



def calc_support_vector_classifier():
    print("*" * 25, "SVM Classification")
    from sklearn.svm import SVC
    model = SVC(kernel='rbf')
    model.fit(X_train, y_train)  # Trained wih 80% Data
    y_pred = model.predict(X_test)
    r21 = r2_score(y_test, y_pred)
    print('lg r2_score:', r21)
    mae = mean_absolute_error(y_test, y_pred)
    print('lg mae:', mae)
    mse = mean_squared_error(y_test, y_pred)
    print('LG mse:', mse)
   
    return r21,mae,mse

def calc_perceptron_classifier():
    ann_model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)  # Output layer for regression
    ])

    # Compile the model
    ann_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train the model
    ann_model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=1)

    # Evaluate the ANN model
    ann_loss, ann_mae = ann_model.evaluate(X_test, y_test)
    return ann_loss, ann_mae


# def test_user_date(test_features):
#     print(test_features)
#     from sklearn.ensemble import RandomForestClassifier
#     from sklearn.svm import SVC
#     model = SVC()
#     model.fit(X_train, y_train)
#     test_pred = model.predict([test_features])
#     return test_pred


def test_user_date(test_features):
    from sklearn.svm import SVR  # Use SVR for regression, not SVC
    
    # Ensure test_features is in the correct format
    num_features = ['AccelSec', 'TopSpeed_KmH', 'Range_Km', 'Battery_Pack Kwh',
                    'Efficiency_WhKm', 'FastCharge_KmH', 'Seats']
    cat_features = ['RapidCharge', 'PowerTrain', 'PlugType', 'BodyStyle', 'Segment']
    
    # Split numerical & categorical features
    test_num = np.array(test_features[:len(num_features)]).reshape(1, -1)
    test_cat = np.array(test_features[len(num_features):]).reshape(1, -1)

    # Apply preprocessing (same as training)
    test_num_scaled = scaler.transform(test_num)
    test_cat_encoded = encoder.transform(test_cat)

    # Combine processed numerical and categorical features
    test_processed = np.hstack((test_num_scaled, test_cat_encoded))

    # Train and predict using SVR instead of SVC
    model = SVR(kernel='rbf')
    model.fit(X_train, y_train)  # Train the model
    test_pred = model.predict(test_processed)  # Predict
    
    return test_pred[0]  # Return the predicted price


def calculate_ann_results():
    from keras.models import Sequential
    from keras.layers import Dense
    classifier = Sequential()
    classifier.add(Dense(output_dim=4, init='uniform', activation='relu', input_dim=5))
    classifier.add(Dense(output_dim=4, init='uniform', activation='relu'))
    classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(classifier.summary())
    classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)

    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)
    accuracy = accuracy_score(y_test, y_pred)
    print('ANN Accuracy:', accuracy)
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    print('ANN Precision Score:', precision)
    recall = recall_score(y_test, y_pred)
    print('ANN Recall Score:', recall)
    f1score = f1_score(y_test, y_pred)
    print('ANN F1-Score:', f1score)
    return accuracy, precision, recall, f1score


