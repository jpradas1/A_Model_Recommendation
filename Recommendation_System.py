import numpy as np
import pandas as pd
from unidecode import unidecode

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate


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


    
    def __init__(self, threshold: int):
        self.threshold = threshold

    def _etl_movie_rating(self):

        # etl for this model
        df_count = df_ratings[['userId','id']].groupby('id').count()
        df_count.reset_index(inplace=True)
        df_count.rename(columns={'userId': 'count'}, inplace=True)
        df_count = df_count.loc[df_count['count'] >= self.threshold]
        df_count['movieId'] = np.arange(1, df_count.shape[0] +1, 1)

        df = pd.merge(df_ratings, df_count, on='id', how='left').dropna()
        return df[['userId','movieId','rating', 'id']]

    def _KNN_movie_rating(self, user: int, title: str,  Knneighbors: int):
        knn = int(np.sqrt(Knneighbors)) + 1
        df = self._etl_movie_rating()
        movies_rating = df.pivot_table(index='movieId', columns='userId', values='rating')\
                        .fillna(0)
        
        # saving efficiently the previous matrix
        movie_rating_matrix = csr_matrix(movies_rating.values)

        # defining and training the model
        model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
        model_knn.fit(movie_rating_matrix)

        # movies_rating.iloc[random_index, :].values.reshape(1,-1)
        records, recom = self._get_sample_user(user, title)

        sample2 = movies_rating.loc[recom, :].values.reshape(1,-1)
        
        distances, indices = model_knn.kneighbors(sample2, n_neighbors=Knneighbors)
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
    def _get_sample_user(self, user: int, title: str):

        # we suppose is the exact title
        title = unidecode(title).lower()

        # records for this user
        df = self._etl_movie_rating()
        records = df.loc[df['userId'] == user, 'movieId'].values

        # movie id to get similarity
        movieid = df_movies.loc[df_movies['title'] == title, 'id'].values[0]
        movieid = df.loc[df['id'] == movieid, 'movieId'].values[0]

        return records, movieid

    def _surprise_recommendation(self, user: int, title: str):

        df_count = df_ratings[['userId','id']].groupby('id').count()
        df_count.reset_index(inplace=True)
        df_count.rename(columns={'userId': 'count'}, inplace=True)

        # The filter on movies is that each movie must count '#threshold'  
        # or more grades
        df_count = df_count.loc[df_count['count'] >= self.threshold]

        df = pd.merge(df_ratings, df_count, on='id', how='left')
        df.dropna(inplace=True)

        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df[['userId', 'id', 'rating']], reader)

        algo = SVD()
        # cross = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
        trainset = data.build_full_trainset()
        algo.fit(trainset)

        movie = df_movies.loc[df_movies['title'] == title, 'id'].values[0]

        prediction = algo.predict(user, movie)[3]

        return prediction
    
    def get_Crecommendation(self, user: int, title: str, similarity: float, Knneighbors: int):
        similarities = self._KNN_movie_rating(user, title, Knneighbors)

        is_greater = similarities > similarity

        if is_greater.any():
            # print("The movie '{}' is recommended for the user '{}'".format(title,user))
            return True
        else:
            # print("The user '{}' may not like the film '{}'".format(user, title))
            return False
        # return matching
    
    def get_Srecomendation(self, user: int, title: str, grade: float):

        # movie = df_movies.loc[df_movies['title'] == title, 'id'].values[0]
        rating = self._surprise_recommendation(user, title)

        print("For the movie '{}' the user '{}' would grade it at {:.2f}".format(title, user, rating))
        if rating >= grade:
            # print('Then the movie is recommended')
            return True
        else:
            # print('Then the movie is not recommended')
            return False
    
    def Cosine_surprise(self, user: int, title: str, similarity: float, grade: float, Knneighbors: int):
        similarities = self._KNN_movie_rating(user, title, Knneighbors)
        v_max = max(similarities)

        rating = self._surprise_recommendation(user, title)

        matching = (v_max + rating/5.0) * 0.5
        limit = (similarity + grade/5.0) * 0.5

        if matching > limit:
            return True
        else:
            return False

# R = Recommendation(rating_w=0.5, genre_w=0.5, threshold=500, Knneighbors=8)
# user = 27833
# title = df_movies.loc[df_movies['id'] == 'ds1', 'title'].values[0]
# print(R.get_Srecomendation(27833, title, matching=3.5))
# print(R.surprise_recommendation(user, title))
# movie = df_movies.loc[df_movies['title'] == '', 'id'].values[0]
# print(title)