import gradio as gr

from Recommendation_System import Recommendation

def main(user, title, matching, rating_w, genre_w, threshold, KNeighbors):
    KNeighbors = int(KNeighbors)
    R = Recommendation(rating_w, genre_w, threshold, KNeighbors)

    return R.get_recommendation(user, title, matching)

inp = ["number", "text",  gr.Slider(0, 1),  gr.Slider(0, 1), gr.Slider(0, 1), "number", "number"]

demo = gr.Interface(
        fn=main, 
        inputs=inp,
        outputs="text",
    )

demo.launch(share=True)