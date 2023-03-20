import gradio as gr
from unidecode import unidecode
import re

from Recommendation_System import Recommendation

def message(title):
    return "The movie {} is not in the dataset, please select another title or \n \
            be sure you wrote it well or decrease the threshold"

def main(user, title, KNN, Surprise, similarity, grade, threshold, KNeighbors):
    # we suppose is the exact title
    title = unidecode(title).lower()
    R = Recommendation(threshold)

    if title in R.get_titles():
        return models(user, title, KNN, Surprise, similarity, grade, 
                      threshold, KNeighbors, R)
    else:
        return message(title)

def models(user, title, KNN, Surprise, similarity, grade, threshold, 
           KNeighbors, R: Recommendation):

    KNeighbors = int(KNeighbors)
    # R = Recommendation(threshold)

    if KNN:
        return R.get_Crecommendation(user, title, similarity, KNeighbors)
    elif Surprise:
        return R.get_Srecomendation(user, title, grade)
    else:
        return R.Cosine_surprise(user, title, similarity, grade, KNeighbors)

inp = [
    "number", 
    "text",
    "checkbox",
    "checkbox",
    gr.Slider(0.3, 1),
    gr.Slider(2,5),
    gr.Slider(300,600), 
    gr.Slider(8,32)
]

demo = gr.Interface(
        fn=main, 
        inputs=inp,
        outputs="text",
    )

demo.launch(share=True)