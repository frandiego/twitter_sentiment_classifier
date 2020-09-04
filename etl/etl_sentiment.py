import gc
import logging
import os

import pandas as pd
from transformers import BertTokenizer, TextClassificationPipeline
from transformers.modeling_auto import BertForSequenceClassification

# ENV VARS
_DATA_PATH = "../data"
_DATA_PATH_NEW = "../sentiment_data"
_DATA_FILE = "sentiment140.csv"
_DATA_FILE_NEW = "sentiment140_bert_{n}.csv"
_HUGGINGFACE_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"
_RANDOM_STATE = 0
_MAX_NUMBER_ROWS = 1_000

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    # FILENAME OF THE DATA
    filename = os.path.join(_DATA_PATH, _DATA_FILE_NEW)
    if os.path.exists(filename):
        logging.info(f'dataset {filename} has been already created.')
    else:
        # READ PREVIOUS DATA
        if not os.path.exists(_DATA_PATH_NEW):
            os.mkdir(_DATA_PATH_NEW)

        df = pd.read_csv(os.path.join(_DATA_PATH, _DATA_FILE))
        index_list = df.index.tolist()
        n_rows = len(index_list)
        batches = [index_list[i:i + _MAX_NUMBER_ROWS] for i in range(0, n_rows, _MAX_NUMBER_ROWS)]

        # DOWNLOAD DE MODEL AND TOKENIZER FROM HUGGINGFACE
        model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=_HUGGINGFACE_MODEL)
        tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=_HUGGINGFACE_MODEL)

        # CREATE THE CLASSIFIER
        sentiment_analyzer = TextClassificationPipeline(model=model, tokenizer=tokenizer)

        # INFER SENTIMENTS
        for i in range(len(batches)):
            # get a portion of data
            df_i = df.loc[batches[i]]
            # create the filename
            filename = os.path.join(_DATA_PATH_NEW, _DATA_FILE_NEW.format(n=str(i).rjust(len(str(len(batches))), '0')))
            if not os.path.exists(filename):
                try:
                    # extract the text clean, but is going to be cleaned again by the tokenizer
                    text = list(df_i['text_clean'].astype(str).values)
                    # infer sentiments
                    sentiments = sentiment_analyzer.transform(text)
                    # STORE IN THE SAME DATAFRAME
                    df_i['sentiment_label'] = list(map(lambda i: i['label'], sentiments))
                    df_i['sentiment_score'] = list(map(lambda i: i['score'], sentiments))
                    df_i['sentiment_label'] = df_i['sentiment_label'].str.replace('stars|star', '').astype(int)
                    # SAVE DATA IN A CSV
                    df_i.to_csv(filename, index=False)
                    logging.info(f'{filename} created.')
                    gc.collect()
                except:
                    pass
