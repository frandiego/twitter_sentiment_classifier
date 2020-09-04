from preprocessor import clean
from nlp import load_dataset

import pandas as pd
import logging
import os


# ENV VARS
_DATA_PATH         = "../data"
_DATASET_NAME      = "sentiment140"
_DATASET_FILES     = ["train", "test"]
_DATA_FILE         = "sentiment140.csv"


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    # FILENAME OF THE DATA
    filename = os.path.join(_DATA_PATH, _DATA_FILE)

    if os.path.exists(filename):
        logging.info(f'dataset {filename} has been already created.')
    else:
        # DOWNLOAD DATASET AND SAVE IN MEMORY USING PANDAS
        dataset = load_dataset(_DATASET_NAME)
        df = pd.concat((dataset[i]._data.to_pandas() for i in _DATASET_FILES)).reset_index().drop_duplicates()

        # SPECIFIC CLEANING FOR TWITTER TEXT : REMOVE URLS, HASHTAGS, MENTIONS,
        # RESERVED WORDS (RT, FAV), EMOJIS AND SMILEYS
        df['text_clean'] = df['text'].apply(clean)

        # SAVE DATASET IN A CSV
        df.to_csv(os.path.join(_DATA_PATH,_DATA_FILE), index=False)

