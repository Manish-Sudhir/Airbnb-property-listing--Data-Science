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
    df2= combine_description_strings(df1)
    dfClean= set_default_values(df2)
    return dfClean

if __name__ == "__main__" :
    df = pd.read_csv('listing.csv')
    dfClean = clean_tabular_data(df)
    dfClean.to_csv('listingClean.csv')


def load_airbnb(label='price_night'):
    # Read the CSV file into a DataFrame
    df = pd.read_csv('listing.csv')
    
    # Filter out columns with text data
    numerical_columns = ['Cleanliness_rating', 'Accuracy_rating', 'Communication_rating',
                         'Location_rating', 'Check-in_rating', 'Value_rating',
                         'guests', 'beds', 'bathrooms', 'price_night']
    df_numerical = df[numerical_columns]
    
    # Remove the label from the features and assign it as the labels
    features = df_numerical.drop(columns=[label])
    labels = df_numerical[label]
    
    return features, labels
