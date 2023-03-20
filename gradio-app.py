import gradio as gr

from Recommendation_System import Recommendation

def find_title(title):
    return None

def main(user, title, KNN, Surprise, similarity, grade, threshold, KNeighbors):
    KNeighbors = int(KNeighbors)
    R = Recommendation(threshold)

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