import sys
import pandas as pd

from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):

    messages=pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df=messages.merge(categories, on='id', how='left')

    return df



def clean_data(df):

    # create a dataframe of the 36 individual category columns and rename
    categories=df['categories'].str.split(';', expand=True)
    categories_colname=categories.loc[0].apply(lambda x: x[:-2]).tolist()
    categories.columns=categories_colname

    # keep only 1, 0 and convert to integer
    for col in categories_colname:
        categories[col]=pd.to_numeric(categories[col].str[-1], downcast='integer')
        # Check number not in (0,1) and update other value to 1
        categories.loc[categories[col] > 1,col] = 1

    # concat to get the new dataframe
    df=pd.concat([df.drop(columns=['categories']), categories], axis=1)

    # remove duplicated records
    df=df.drop_duplicates()

    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///{}'.format(database_filename))

    # connection = engine.raw_connection()
    # cursor = connection.cursor()
    # command = "DROP TABLE IF EXISTS InsertTableName;"
    # cursor.execute(command)
    # command = "DROP TABLE IF EXISTS Message;"
    # cursor.execute(command)
    # print (engine.table_names())

    df.to_sql('message_categories',engine,index=False,if_exists='replace')

    # print(engine.table_names())
    # connection.commit()
    # cursor.close()


def main():

    if len(sys.argv) == 4:
        print(sys.argv[1:])
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

    # for debug
    # else:
    #     messages_filepath='disaster_messages.csv'
    #     categories_filepath='disaster_categories.csv'
    #     database_filepath='DisasterResponse.db'


        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
            .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        df.head()
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