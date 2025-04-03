import os
import sys
sys.path.append(os.path.abspath('..'))

from src import config
from src.load_data import load_data
from src.make_model import train_model_1, train_model_2

import logging
# Set up logging
logging.basicConfig(filename='../logs/pipeline.log', level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    # carico il dataset: load_data.py
    logging.info("Starting pipeline...")
    load_data()

    # preprocess: i dati non hanno bisogno di preprocess

    # creo il modello: make_model.py
    logging.info("Training the model...")
    train_model_1()
    train_model_2()

if __name__ == "__main__":
    main()