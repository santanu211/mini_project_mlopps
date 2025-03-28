import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import yaml

def load_params(path):
    test_size=yaml.safe_load(open(path,"r"))["data"]["dataload"]["test_size"]
    return test_size


def load_data(data_url:str)->pd.DataFrame:
    df=pd.read_csv(data_url)

    return df
def preprosed(df:pd.DataFrame)->pd.DataFrame:
    # 'id' is just an identifier, and 'City', 'Profession', 'Degree' may not be useful
    df = df.drop(columns=['id', 'City', 'Profession', 'Degree'], errors='ignore')

    # 3️⃣ Handle missing values
    # Fill missing numerical values with the mean
    df.fillna(df.mean(numeric_only=True), inplace=True)
    return df


def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str)->None:
    data_path=os.path.join(data_path,"raw")
    os.makedirs(data_path,exist_ok=True)
    train_data.to_csv(os.path.join(data_path,"train.csv"),index=False)
    test_data.to_csv(os.path.join(data_path,"test.csv"),index=False)

def main():
    test_size=load_params("C:/Users/Admin/Music/mlop1/depressed_or_not/params.yaml")
    df = load_data("C:/Users/Admin/Music/Student Depression Dataset.csv")

    final_df=preprosed(df)
    train_data,test_data=train_test_split(final_df,test_size=test_size, random_state=42)
    save_data(train_data, test_data, data_path='C:/Users/Admin/Music/mlop1/depressed_or_not/datas')
if __name__=="__main__":
    main()



