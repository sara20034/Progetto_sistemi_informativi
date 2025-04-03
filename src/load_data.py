import os
import sys
import sqlite3
import pandas as pd

sys.path.append(os.path.abspath('..'))  # Adds the parent directory to sys.path
from src import config

import logging

def load_data():
    logging.info('Opening Excel File...')
    df = pd.read_excel(os.path.join(config.RAW_DATA_PATH, 'Real estate valuation data set.xlsx'))

   # Create a connection to the SQLite database (or create if it doesn't exist)
    conn = sqlite3.connect(config.DATABASE_PATH)

    # Write the DataFrame to a table (replace 'my_table' with your desired table name)
    df.to_sql(config.RAW_TABLE, conn, if_exists='replace', index=False)

    # Commit and close the connection
    conn.commit()
    conn.close()

    logging.info(f"Data successfully written to {config.RAW_TABLE} table.")