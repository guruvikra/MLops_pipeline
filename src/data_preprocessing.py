import pandas as pd
import os
import logging
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.stem.porter import  PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download("punkt")
from nltk.tokenize import WordPunctTokenizer
import string

logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../logs")
logs_dir = os.path.abspath(logs_dir)  # Convert to absolute path
os.makedirs(logs_dir, exist_ok=True)

# createing a object for logging
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(logs_dir, 'data_pre_processing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


logger.addHandler(console_handler)
logger.addHandler(file_handler)

tokenizer = WordPunctTokenizer()



def transform_text(text):
    """
    Transforms the input text by converting it to lowercase, tokenizing, removing stopwords and punctuation, and stemming.
    """
    ps = PorterStemmer()
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    text = tokenizer.tokenize(text)
    # Remove non-alphanumeric tokens
    text = [word for word in text if word.isalnum()]
    # Remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    # Stem the words
    text = [ps.stem(word) for word in text]
    # Join the tokens back into a single string
    return " ".join(text)


def preprocess_data(df, text_col='text',target_col="target"):
    """
    Preprocesses the text data in the given DataFrame by applying the transform_text function.
    """
    try:
        # Log the start of preprocessing
        logger.info("Starting data preprocessing...")
        encoder = LabelEncoder()

        df[target_col] =  encoder.fit_transform(df[target_col])
        logger.debug("encoded")
        # Log the end of preprocessing
        logger.info("Data preprocessing completed.")

        df = df.drop_duplicates(keep = 'first')

        # df[text_col] = df[text_col].apply(transform_text)
        df.loc[:,text_col] = df[text_col].apply(transform_text)
        logger.debug("text tranformed")
        logger.debug(df.head())
        return df
    except Exception as e:
        # Log any errors during preprocessing
        logger.error(f"An error occurred during preprocessing: {str(e)}")
        raise e

def main():
    """"main function"""
    try:
        train_data = pd.read_csv('/home/namlabs/Guru/DVC/Spam-detection/MLops_pipeline/data/raw/train.csv')
        test_data = pd.read_csv('/home/namlabs/Guru/DVC/Spam-detection/MLops_pipeline/data/raw/test.csv')
        logger.debug("data loaded")

        train_data_processed = preprocess_data(train_data)
        test_data_processed = preprocess_data(test_data)
        logger.debug("data preprocessed")

        data_path = os.path.join('./data',"interim")
        os.makedirs(data_path, exist_ok=True)

        train_data_processed.to_csv(os.path.join(data_path, "train_preprocessed.csv"), index=False)
        test_data_processed.to_csv(os.path.join(data_path, "test_preprocessed.csv"), index=False)

        logger.info("Data saved to interim folder.")
    except Exception as e:
    # Log any errors during data saving
        logger.error(f"An error occurred during data saving: {str(e)}")
        raise e

if __name__ == "__main__":
    main()