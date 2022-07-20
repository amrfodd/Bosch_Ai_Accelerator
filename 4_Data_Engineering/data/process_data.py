import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loading the separated csv files and merge them in one dataframe using pandas
  
    Parameters:
    messages_filepath: location of the message dataset to be loaded using pandas
    categories_filepath: location of the categories dataset to be loaded using pandas
    
    Returns:
    df: merged dataframe of the previous datasets  
    """
    # Read messages dataset
    messages = pd.read_csv(messages_filepath)
    
    #read categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge the two previous dataframes
    df = pd.merge(messages, categories, how = 'left', on = 'id')
        
    return df
    
def clean_data(df):
    """
    This function is intended to clean the dataframe. i.e(Clean columns, remove duplicates, remove inconsistency.
  
    Parameters:
    df: The dataframe which is merge of the messages and categories CSV files
    
    Returns:
    df: cleaned data
  
    """
    ## make dataframe for the expanded categories column in the df datafrme
    categories = df.categories.str.split(';', expand=True)
    row = categories.iloc[0]

    
    # Preprocess the new category dataframe
    category_colnames = []
    for text in row:
        category_colnames.append(text.split('-')[0])

    categories.columns = category_colnames

    for column in categories:
    # set each value to be the last character of the string
        categories[column] = [x.split('-')[1] for x in categories[column]]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # Replace rows containing the value 2     
    categories.replace(2, 1, inplace=True)
    
    # Drop the main categoris column
    df.drop(['categories'], axis = 1, inplace = True)
    
    # Concat the dataframe and the expanded categories dataframe
    df = pd.concat([df, categories], sort = False, axis = 1)
    # Drop duplicates
    df = df.drop_duplicates()
    
    return df

def save_data(df, database_filename):
    """
    This Function used to save the data in a Databse
  
  
    Parameters:
    df: The cleaned dataframe to be saved in a databse
    database_filename: Name assigned to the databse
    
    Returns:
    DataBase
  
    """
    
    # Save the dataframe as a database
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterProcess', engine, index=False, if_exists='replace')

    
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()