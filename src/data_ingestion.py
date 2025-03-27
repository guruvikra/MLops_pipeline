import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging
import yaml

# here is the location of the log files
# Ensure logs directory is outside the "src" folder
project_root = os.getcwd()
logs_dir = os.path.join(project_root, "../logs")
os.makedirs(logs_dir, exist_ok=True)

# createing a object for logging
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(logs_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


logger.addHandler(console_handler)
logger.addHandler(file_handler)



def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


def load_data(file_url: str) -> pd.DataFrame:
    """Load data from the csv file."""
    try:
        df = pd.read_csv(file_url)
        logger.debug("data loaded from %s" , file_url)
        return df
    except Exception as e:
        logger.error("%s", e)
        raise

def preprocess_data(df: pd.DataFrame)-> pd.DataFrame:
    """Preprocess the data."""
    try:
        df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'], inplace = True)
        df.rename(columns= {'v1': 'target' , 'v2': 'text'}, inplace = True)
        logger.debug('preprocessing completed')
        return df
    except Exception as e:
        logger.error("%s", e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame , url: str) -> None:
    """Save the preprocessed data to a csv file."""
    try:
        data_path = os.path.join(url, 'raw')
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path,'train.csv'), index=False)
        test_data.to_csv(os.path.join(data_path,'test.csv'), index=False)
        logger.debug('data saved to %s and %s' , url + '_train.csv', url + '_test.csv')
        logger.debug('data saved to %s ', data_path)
    except Exception as e:
        logger.error("%s", e)
        raise



def main():
    try:
        # test_size = 0.2
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']
        data_path = '/home/namlabs/Guru/DVC/Spam-detection/MLops_pipeline/experiments/spam.csv'
        df = load_data(file_url=data_path)
        df_preprocessed = preprocess_data(df=df)
        train_data, test_data = train_test_split(df_preprocessed, test_size=test_size, random_state=42)
        save_data(train_data=train_data, test_data=test_data, url ='./data')
    except Exception as e:
        logger.error("%s", e)


if __name__ == "__main__":
    main()