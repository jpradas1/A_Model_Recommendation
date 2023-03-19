import numpy as np
import pandas as pd
from unidecode import unidecode

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity


import warnings
warnings.filterwarnings('ignore')

from ETL import ETL_class

path_rating = './dataset/ratings/'
path_titles = './dataset/titles/'
etl = ETL_class(path_titles, path_rating)

df_movies = etl.get_movies()[['id', 'title', 'listed_in']] \
            .rename(columns={'listed_in': 'genre'})

df_ratings = etl.get_ratings()[['userId', 'rating', 'movieId']] \
            .rename(columns={'movieId':'id'})

class Recommendation(object):
    
    def __init__(self, rating_w: float, genre_w: float, threshold: int, Knneighbors: int):
        self.threshold = threshold
        self.rating_w = rating_w
        self.genre_w = genre_w
        self.Knneighbors = Knneighbors

    def etl_movie_rating(self):

        # etl for this model
        df_count = df_ratings[['userId','id']].groupby('id').count()
        df_count.reset_index(inplace=True)
        df_count.rename(columns={'userId': 'count'}, inplace=True)
        df_count = df_count.loc[df_count['count'] >= self.threshold]
        df_count['movieId'] = np.arange(1, df_count.shape[0] +1, 1)

        df = pd.merge(df_ratings, df_count, on='id', how='left').dropna()
        return df[['userId','movieId','rating', 'id']]

    def KNN_movie_rating(self, user: int, title: str):
        knn = int(np.sqrt(self.Knneighbors)) + 1
        df = self.etl_movie_rating()
        movies_rating = df.pivot_table(index='movieId', columns='userId', values='rating')\
                        .fillna(0)
        
        # saving efficiently the previous matrix
        movie_rating_matrix = csr_matrix(movies_rating.values)

        # defining and training the model
        model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
        model_knn.fit(movie_rating_matrix)

        # movies_rating.iloc[random_index, :].values.reshape(1,-1)
        records, recom = self.get_sample_user(user, title)

        sample2 = movies_rating.loc[recom, :].values.reshape(1,-1)
        
        distances, indices = model_knn.kneighbors(sample2, n_neighbors=self.Knneighbors)
        idx_sort = np.argsort(distances[0])[::-1]
        indices = [indices[0][ii] for ii in idx_sort]

        most_similar_samples = [movies_rating.loc[indices[x]].values.reshape(1,-1) for x in range(knn)]

        similarity = []
        for r in records:
            sample1 = movies_rating.loc[r, :].values.reshape(1,-1)
            MSS = [cosine_similarity(sample1, mss)[0][0] for mss in most_similar_samples]
            similarity.append(MSS)

        similarity = np.array(similarity).flatten()

        return similarity
    
    # get the movies watched by the user
    def get_sample_user(self, user: int, title: str):

        # we suppose is the exact title
        title = unidecode(title).lower()

        # records for this user
        df = self.etl_movie_rating()
        records = df.loc[df['userId'] == user, 'movieId'].values

        # movie id to get similarity
        movieid = df_movies.loc[df_movies['title'] == title, 'id'].values[0]
        movieid = df.loc[df['id'] == movieid, 'movieId'].values[0]

        return records, movieid

    def get_recommendation(self, user: int, title: str, matching = 0.6):
        similarities = self.KNN_movie_rating(user, title)

        is_greater = similarities > matching

        if is_greater.any():
            # print("The movie '{}' is recommended for the user '{}'".format(title,user))
            return True
        else:
            # print("The user '{}' may not like the film '{}'".format(user, title))
            return False
        # return matching
