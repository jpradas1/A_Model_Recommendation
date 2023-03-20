# Model of Recommendations

This repository displays through [gradio-app](https://github.com/gradio-app/gradio) two machine learning model applied in python for a movie-recomendation system. Both models aim at whether or not recommend a movie to a specific user, the first one is based on the Cosine Similarity Method and the second one employs the python library [surprise](https://github.com/NicolasHug/Surprise) build as a machine learning model to perform recommentacions.

Besides, this project guides throughout the different stage for consuming the final data, i.e., it shows the ETL, EDA and ML deployment needed to get the recommendation system.

## ETL & Deploy on FastAPI
> The project starts by cleasing the dataset avaible formed by csv files which contains informaiton of 22998 movies on diferent streaming platform such Amazon Prime, Disney, Hulu and Netflix, there's data about users and their rating for some movies.
>
> Each step for the data cleansing is located in the file [ETL.py](https://github.com/jpradas1/A_Model_Recommendation/blob/main/ETL.py), where the code is oriented to normalized all text by converting it in lowercase no exception. In the column rating missing values was replaced by 'g' (general audience) as well.
>
> Once the data has been cleaning, we have created an API using FastAPI framework to allows users consum data available, thanks to [render.com](https://render.com/), at [FastAPI deploy](https://a-model-recommendation.onrender.com/docs), so there anyone can perform queries on cleaned data.
>
> <img src=".src/fastapi.png" width="800">
### How to run it
> The stable version to run the api is in the branch [fast-api](https://github.com/jpradas1/A_Model_Recommendation/tree/fast-api), here is the way to perform a suitable running.
## EDA
> 
