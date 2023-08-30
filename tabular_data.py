import pandas as pd
df = pd.read_csv('listing.csv')

def remove_rows_with_missing_rating(df):
    df2=df.dropna(subset=['Cleanliness_rating','Accuracy_rating','Communication_rating', 'Location_rating','Check-in_rating','Value_rating'])
    return df2

def combine_description_strings(df):
    df['Description'] = df['Description'].str.replace(r'\s+', ' ')
    df['Description'] = df['Description'].str.replace(r'About this space', '')
    return df

def set_default_values(df):
    df['guests'] = df['guests'].fillna(1)
    df['beds'] =  df['beds'].fillna(1)
    df['bathrooms'] = df['bathrooms'].fillna(1)
    return df

def clean_tabular_data(df):
    df1= remove_rows_with_missing_rating(df)

    # Convert textual entries to 1 in numerical columns
    df2= combine_description_strings(df1)
    dfClean= set_default_values(df2)
    return dfClean

if __name__ == "__main__" :
    df = pd.read_csv('listing.csv')
    dfClean = clean_tabular_data(df)
    dfClean.to_csv('listingClean.csv')


def load_airbnb(label):
    # Read the CSV file into a DataFrame
    df = pd.read_csv('listing.csv')
    df_cleaned = clean_tabular_data(df)

    # Filter out columns with text data
    numerical_columns = ['Cleanliness_rating', 'Accuracy_rating', 'Communication_rating',
                         'Location_rating', 'Check-in_rating', 'Value_rating',
                         'guests', 'beds', 'bathrooms', 'Price_Night']
    df_numerical = df_cleaned[numerical_columns]

    # Convert textual entries to 1 in numerical columns
    # df_cleaned = df_numerical.copy()
    # for column in numerical_columns:
    #     df_cleaned[column] = pd.to_numeric(df_cleaned[column], errors='coerce').fillna(1)
    # df_cleaned = df_numerical[~df_numerical['guests'].str.contains('Somerford Keynes England Unit')]
    df_cleaned = df_numerical[~df['guests'].apply(lambda x: isinstance(x, str) and 'Somerford Keynes England Unit' in x)]


    
    # Remove the label from the features and assign it as the labels
    features = df_cleaned.drop(columns=[label])
    labels = df_cleaned[label]
    
    return features, labels
