import joblib
import numpy as np
import streamlit as st

def pred(model, a, b, c, d):
    res = []
    for x in np.arange(-5.0, 5.0, 0.1):
        for y in np.arange(-5.0, 5.0, 0.1):
            res.append(model.predict([[x, y, a, b, c, d]]))
    return np.array(res).reshape([100, 100])

with open("model.pkl", "rb") as f:
    model = joblib.load(f)

a = st.sidebar.slider("a", 0.0, 1.0, 0.5, 0.01)
b = st.sidebar.slider("b", -1.0, 1.0, 0.0, 0.01)
c = st.sidebar.slider("c", 0.0, 1.0, 0.5, 0.01)
d = st.sidebar.slider("d", -1.0, 1.0, 0.0, 0.01)

img_array = pred(model, a, b, c, d)
img_array_min0max1 = (img_array - img_array.min()) / (img_array.max() - img_array.min())

st.image(image=img_array_min0max1)