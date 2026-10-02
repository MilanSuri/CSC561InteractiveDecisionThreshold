import streamlit as st
import log_reg

st.title("CSC 561 Interactive Decision Threshold:")
st.write("Adjust the decision threshold to see how it affects the model's classification results.")

X, y = log_reg.get_data()
model, X_test, y_test = log_reg.train_log_reg(X, y)

threshold = st.slider ("Decision Threshold Slider:", min_value= 0.0, max_value = 1.0, step = 0.01)

probabilities, predictions = log_reg.predictions(model, X_test, threshold)

confusion_matrix = log_reg.generate_confusion_matrix(y_test, predictions)
tn, fp, fn, tp = confusion_matrix.ravel()

st.subheader("Classification Results")

col1, col2, col3, col4 = st.columns(4)

col1.metric("True Positives", tp)
col2.metric("True Negatives", tn)
col3.metric("False Positives", fp)
col4.metric("False Negatives", fn)