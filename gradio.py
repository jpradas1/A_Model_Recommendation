from gradio import Interface

from Recommendation_System import Recommendation

def main(user, title, matching, rating_w, genre_w, threshold, KNeighbors):
    R = Recommendation(rating_w, genre_w, threshold, KNeighbors)

    return R.get_recommendation(user, title, matching)

integer = {"type": "number", "label": "Integer input"}
floater = {"type": "number", "label": "Float input"}
strer = {"type": "text", "label": "String input"}

input = [integer, strer, floater, floater, floater, integer, integer]
demo = Interface(fn=main, inputs=input, outputs="text")
demo.launch()