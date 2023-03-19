from fastapi import FastAPI

import numpy as np
import pandas as pd
import re
import os
from unidecode import unidecode

app = FastAPI()

def get_csv_files(path):
    raw_files = os.listdir(path)
    file = []
    for item in raw_files:
        p = re.findall("(\w*\.csv)", item)
        if p:
            file.append(p[0])
    return np.sort(file)

path_titles = './dataset/'

df_movies = []
for t in get_csv_files(path_titles):
    df_movies.append(pd.read_csv(path_titles + t))

df_movies = pd.concat(df_movies)

# last_year = np.sort(df_movies['release_year'].unique())[-1]

@app.get("/function1/{year},{platform},{duration_type}")
def get_max_duration(year: int, platform: str, duration_type: str):
    duration_type = unidecode(duration_type).lower()
    platform = unidecode(platform).lower()
    
    df = df_movies[['id','release_year','duration_int','duration_type']]

    begins = platform[0]
    df = df.loc[(df['duration_type'] == duration_type) & (df['release_year'] == year)]
    df = df.loc[df['id'].str.contains('^{}'.format(begins))]

    idx = df.loc[df.duration_int == df.duration_int.max()].id.values[0]

    row = df_movies.loc[df_movies['id'] == str(idx)][['id','title', 'duration_int', 'duration_type']]
    column = row.columns

    result = {c: v for (c,v) in zip(column, row.values[0])}
    # print("The longest movie or series in duration [{}] is '{}', it lasts {} {}".format(duration_type, row['title'].values[0],row['duration_int'].values[0],duration_type))
    return result

# print(get_max_duration(2018, 'amazon', 'min'))

@app.get("/function2/{platform},{scored},{year}")
def get_score_count(platform: str, scored: float, year: int):
    platform = unidecode(platform).lower()
    
    begins = platform[0]
    df = df_movies[['id', 'release_year', 'mean_rating']]
    df = df.loc[df.release_year == year]
    df = df.loc[df['id'].str.contains('^{}'.format(begins))]
    
    # merge = df.merge(df_ratings[['movieId', 'mean_rating']], left_on='id',right_on='movieId')
    # merge = merge[['id', 'mean_rating']].groupby(['id']).agg('mean')

    result = df[df.mean_rating >= scored].shape[0]
    # return result
    return 'result: {}'.format(result)

print(get_score_count('amazon', 2.5, 2006))

@app.get("/function3/{platform}")
def get_count_platform(platform: str):
    # if platform == None:
    #     result = df_movies[['id']].drop_duplicates()
    #     result = df_movies.shape[0]
    #     return result
    # else:
    platform = unidecode(platform).lower()
    begins = platform[0]
    df = df_movies.loc[df_movies['id'].str.contains('^{}'.format(begins))]
    result = df.shape[0]
    return result

@app.get("/function4/{platform},{year}")
def get_actor(platform: str, year: int):
    begins = platform[0]
    df = df_movies[['id','cast', 'release_year']].loc[df_movies.release_year == year]
    df = df.loc[df.id.str.contains('^{}'.format(begins))]
    
    actors = [a for x in df.cast.values for a in str(x).strip( ).split(', ')]
    actors = [unidecode(x).lower() for x in actors]
    
    values, counts = np.unique(actors, return_counts=True)
    cast = {v: int(c) for v, c in zip(values,counts)}
    
    sorted_cast = dict(sorted(cast.items(), key=lambda item: item[1], reverse=True))
    max_keys = {k: v for k, v in list(sorted_cast.items())[:5]}
    
    return max_keys