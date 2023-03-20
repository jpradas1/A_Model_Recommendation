# Model for Recommendations (Deployment)

This API works for executing 4 type of queries on the database about movies, their average ratings, titles, cast and so no. The database has information on 4 streaming platform, i.e., 'Amazon Prime' 'Disney Plus', 'Hulu' and 'Netflix'. The first funciton returns the longest movie in duration by platform and platform. The second one tells us what is the amount of movies which have obtained an average rating greater than a score by year. The third one gives us the number of movies by platform. And the last one, returns the most frequent or common actor by platform and year.

## Local running
Within of a python virtual environtment, we install the first libraries
```
python -m venv name-env/
pip install uvicorn fastapi
```
create the needed file for storing the libraries to run the api
```
pip freeze > requirements.txt
```
Finally we run
```
uvicorn main:app --reload
```
and the api run in the localhost on port 8000 by default, i.e., (http://127.0.0.1:8000/docs).
## On-Line
This particular api is located on [FastAPI deploy](https://a-model-recommendation.onrender.com/docs) which run on the [render.com](https://render.com/) serves.
