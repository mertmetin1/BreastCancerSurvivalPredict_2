import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import joblib
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV




def split_data(df, target_column,test_size=0.2,random_state=42):
    """Split data into features and target variable."""
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)







def train_linear_regression(X_train, y_train, X_test, y_test):
    print("\nLineer Regresyon")
    
    # Modeli oluşturma
    lin_reg = LinearRegression()
    
    # Modeli eğitme
    lin_reg.fit(X_train, y_train)

    # Test seti üzerinde tahmin yapma
    y_test_pred_lin = lin_reg.predict(X_test)

    # Test seti doğruluk skorunu hesaplama (R^2 skoru)
    test_r2 = r2_score(y_test, y_test_pred_lin)
    print("Test Seti R^2 Skoru:", test_r2)

    # Ortalama Kare Hatası hesaplama
    mse_test = mean_squared_error(y_test, y_test_pred_lin)
    print("Test Seti Ortalama Kare Hatası:", mse_test)

    # Modeli kaydet
    joblib.dump(lin_reg, 'models/linear_regression_model.joblib')





def train_logistic_regression(X_train, y_train, X_test, y_test):
    print("Standard Logistic Regression")
    
    log_reg = LogisticRegression(penalty='none', max_iter=1000)  # No regularization
    log_reg.fit(X_train, y_train)

    # Test seti üzerinde tahmin yapma
    y_test_pred = log_reg.predict(X_test)

    # Test seti doğruluk skorunu hesaplama
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print("Test Seti Doğruluk Skoru:", test_accuracy)

    # Modeli kaydet
    joblib.dump(log_reg, 'models/logistic_regression_model.joblib')

def train_logistic_regression(X_train, y_train, X_test, y_test):
    print("\nLogistic Regression")
    
    log_reg = LogisticRegression( max_iter=1000)
    log_reg.fit(X_train, y_train)

    # Test seti üzerinde tahmin yapma
    y_test_pred = log_reg.predict(X_test)

    # Test seti doğruluk skorunu hesaplama
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print("Test Seti Doğruluk Skoru:", test_accuracy)

    # Modeli kaydet
    joblib.dump(log_reg, 'models/logistic_regression_model.joblib')

def train_logistic_regression_lasso(X_train, y_train, X_test, y_test):
    print("\nL1 Regularizasyon (Lasso) ile Logistic Regression")
    
    log_reg_lasso = LogisticRegression(penalty='l1', C=0.1, solver='saga', max_iter=1000)
    log_reg_lasso.fit(X_train, y_train)

    # Test seti üzerinde tahmin yapma
    y_test_pred = log_reg_lasso.predict(X_test)

    # Test seti doğruluk skorunu hesaplama
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print("Test Seti Doğruluk Skoru:", test_accuracy)

    # Modeli kaydet
    joblib.dump(log_reg_lasso, 'models/logistic_regression_lasso_model.joblib')







def train_logistic_regression_ridge(X_train, y_train, X_test, y_test):
    print("\nL2 Regularizasyon (Ridge) ile Logistic Regression")
    
    log_reg_ridge = LogisticRegression(penalty='l2', C=0.1, max_iter=1000)
    log_reg_ridge.fit(X_train, y_train)

    # Test seti üzerinde tahmin yapma
    y_test_pred = log_reg_ridge.predict(X_test)

    # Test seti doğruluk skorunu hesaplama
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print("Test Seti Doğruluk Skoru:", test_accuracy)

    # Modeli kaydet
    joblib.dump(log_reg_ridge, 'models/logistic_regression_ridge_model.joblib')












def train_support_vector_classifier(X_train, y_train, X_test, y_test):
    print("\nSupport Vector Classifier")
    
    svm_model = SVC()
    svm_model.fit(X_train, y_train)
    y_test_pred_svm = svm_model.predict(X_test)
    
    test_accuracy_svm = accuracy_score(y_test, y_test_pred_svm)
    print("Test Seti Doğruluk Skoru (SVM):", test_accuracy_svm)

    # Modeli kaydet
    joblib.dump(svm_model, 'models/svm_model.joblib')






def train_random_forest_classifier(X_train, y_train, X_test, y_test):
    print("\nRandom Forest Classifier")
    
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    y_test_pred_rf = rf_model.predict(X_test)
    
    test_accuracy_rf = accuracy_score(y_test, y_test_pred_rf)
    print("Test Seti Doğruluk Skoru (Random Forest):", test_accuracy_rf)

    # Modeli kaydet
    joblib.dump(rf_model, 'models/rf_model.joblib')










def optimize_svc(X_train, y_train, X_test, y_test):
    print("\nSupport Vector Classifier with Hyperparameter Optimization")
    
    # Define the parameter grid for SVC
    param_grid_svc = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }

    # Initialize Grid Search
    grid_search_svc = GridSearchCV(SVC(), param_grid_svc, scoring='accuracy', cv=5)
    grid_search_svc.fit(X_train, y_train)

    # Best parameters and score
    best_params_svc = grid_search_svc.best_params_
    best_accuracy_svc = grid_search_svc.best_score_

    print("Best Parameters (SVC):", best_params_svc)
    print("Best Cross-Validation Accuracy (SVC):", best_accuracy_svc)

    # Test the best model on the test set
    best_svc_model = grid_search_svc.best_estimator_
    y_test_pred_svc_optimized = best_svc_model.predict(X_test)
    test_accuracy_svc_optimized = accuracy_score(y_test, y_test_pred_svc_optimized)
    print("Test Set Accuracy (Optimized SVC):", test_accuracy_svc_optimized)

    # Modeli kaydet
    joblib.dump(best_svc_model, 'models/optimized_svm_model.joblib')






def optimize_rf(X_train, y_train, X_test, y_test):
    print("\nRandom Forest Classifier with Hyperparameter Optimization")
    
    # Define the parameter grid for RFC
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }

    # Initialize Grid Search
    grid_search_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, scoring='accuracy', cv=5)
    grid_search_rf.fit(X_train, y_train)

    # Best parameters and score
    best_params_rf = grid_search_rf.best_params_
    best_accuracy_rf = grid_search_rf.best_score_

    print("Best Parameters (RFC):", best_params_rf)
    print("Best Cross-Validation Accuracy (RFC):", best_accuracy_rf)

    # Test the best model on the test set
    best_rf_model = grid_search_rf.best_estimator_
    y_test_pred_rf_optimized = best_rf_model.predict(X_test)
    test_accuracy_rf_optimized = accuracy_score(y_test, y_test_pred_rf_optimized)
    print("Test Set Accuracy (Optimized RFC):", test_accuracy_rf_optimized)

    # Modeli kaydet
    joblib.dump(best_rf_model, 'models/optimized_rf_model.joblib')


