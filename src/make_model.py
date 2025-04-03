from src import config
import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import sys
sys.path.append(os.path.abspath('..'))  # Adds the parent directory to sys.path

import pickle
import logging

def load_data():
    """Loads data from the SQLite database."""
    conn = sqlite3.connect(config.DATABASE_PATH)
    query = f"SELECT * FROM {config.PROCESSED_TABLE}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def train_model_1(grid_search=False):
    """Crea un modello di regressione lineare dai dati"""

    # Save original indices 
    df = load_data()
    df_indices = df.index

    # Feature extraction
    X = df[['X5 latitude', 'X6 longitude']]
    y = df['Y house price of unit area']


    # Train-test split (preserve indices)
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, df_indices, test_size=0.2, random_state=42
    )

    if grid_search:
            model = LinearRegression()
            param_grid = {
                'fit_intercept': [True, False],
                'degree': [1, 2, 3],
            }

            grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
        
    else:
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # Salviamo modello in un file
    logging.info('Saving model...')
    with open(os.path.join(config.MODELS_PATH, "LinReg_default.pickle"), "wb") as file:
        pickle.dump(model, file)

    # CDataframe con previsioni
    test_df = df.loc[test_idx].copy() 
    test_df['prediction'] = y_pred

    # Connect to the database
    conn = sqlite3.connect(config.DATABASE_PATH)

    # saving predictions
    test_df.to_sql(config.PREDICTIONS_TABLE, conn, if_exists='replace', index=False)
    
    # Commit and close the connection
    conn.commit()
    conn.close()

def train_model_2(grid_search=False):
    """Crea un modello di regressione lineare dai dati"""
    df = load_data()
    # Save original indices 
    df_indices = df.index

    # Feature extraction
    X = df[['X1 transaction date', 'X2 house age',
       'X3 distance to the nearest MRT station',
       'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']]
    y = df['Y house price of unit area']


    # Train-test split (preserve indices)
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, df_indices, test_size=0.2, random_state=42
    )

    if grid_search:
            model = LinearRegression()
            param_grid = {
                'fit_intercept': [True, False],
                'degree': [1, 2, 3],
            }

            grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
        
    else:
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # Salviamo modello in un file
    logging.info('Saving model...')
    with open(os.path.join(config.MODELS_PATH, "LinReg_extras.pickle"), "wb") as file:
        pickle.dump(model, file)

    # CDataframe con previsioni
    test_df = df.loc[test_idx].copy() 
    test_df['prediction'] = y_pred

    # Connect to the database
    conn = sqlite3.connect(config.DATABASE_PATH)

    # saving predictions
    test_df.to_sql(config.PREDICTIONS_TABLE, conn, if_exists='replace', index=False)
    
    
    # Commit and close the connection
    conn.commit()
    conn.close()