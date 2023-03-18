import numpy as np
import pandas as pd
import re
import os

class ETL_class(object):

    def __init__(self, path_titles: str, path_rating: str):
        self.path_title = path_titles
        self.path_rating = path_rating

    def get_csv_files(self, path):
        raw_files = os.listdir(path)
        file = []
        for item in raw_files:
            p = re.findall("(\w*\.csv)", item)
            if p:
                file.append(p[0])
        return np.sort(file)

    def etl_movies(self, platform: str):
        df = pd.read_csv(self.path_title + platform)
        
        # we create an id for each platform's movie
        df['id'] = str(platform[0]) + df['show_id']

        # fill nan values by G, which depicts “general for all audiences”
        df['rating'].fillna('G', inplace=True)

        # transform datae_added column to YY-MM-DD date format
        df['date_added'] = pd.to_datetime(df['date_added'])

        # come every register into lowercases
        df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

        # here we create two new columns to split duration data into numerical and type data
        df['duration_int'] = df[['duration']].replace('\D*', '', regex=True).astype(float)
        df['duration_type'] = df[['duration']].replace('\d*\s', '', regex=True)

        # let's normilize data in duration_type columns
        df['duration_type'].replace('seasons', 'season', inplace=True)

        columns = ['id', 'show_id', 'type', 'title', 'director', 'cast', 'country', 
                'date_added', 'release_year', 'rating', 'duration', 'duration_int',
                'duration_type', 'listed_in', 'description']
        df = df.reindex(columns=columns)
        return df

    # def get_movies(self):
    #     df_titles = []
    #     titles = self.get_csv_files(self.path_title)
    #     for platform in titles:
    #         df_titles.append(self.etl_movies(platform))
            
    #     df_titles = pd.concat(df_titles)
    #     return df_titles

    # def get_ratings(self):
    #     df_rating = []
    #     rating = self.get_csv_files(self.path_rating)
    #     for dc in rating:
    #         df_rating.append(pd.read_csv(self.path_rating + dc))

    #     df_rating = pd.concat(df_rating)
    #     df_rating.drop_duplicates(inplace=True)
    #     return df_rating
    
path_titles = './dataset/titles/'
path_rating = './dataset/ratings/'
etl = ETL_class(path_titles, path_rating)

titles = etl.get_csv_files(path_titles)
rating = etl.get_csv_files(path_rating)

for t in titles:
    df_t = etl.etl_movies(t)[['id', 'cast', 'release_year', 'duration_int', 'duration_type']]
    df_t.to_csv(path_titles + t, index=False)

for r in rating:
    df_r = pd.read_csv(path_rating + r)[['movieId', 'rating']]
    df_r.to_csv(path_rating + r, index=False)

print(titles)
print(rating)