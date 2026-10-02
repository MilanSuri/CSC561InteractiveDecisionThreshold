'''
Milan Suri
CSC 561
Interactive Decision Threshold

This file is for creating the interactive visual that people interested in ML could use to learn about decision boundaries.
We use the framework from log_reg.py and then implement it for getting clear insights into the distrubtion of results
and their accuracies. Then we also plot the points and adjust the boundary to show how adjusting the threshold changes
the number of false or true postivies/negatives.
'''


import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

import log_reg

# Add title and description
st.title("CSC 561 Interactive Decision Threshold")

st.write("Adjust the decision threshold to see how it affects the model's classification results.")

# Setup the log reg model
X, y = log_reg.get_data()

model, X_test, y_test = log_reg.train_log_reg(X, y)

threshold = st.slider(
    "Decision Threshold Slider:",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.01
)

probabilities, predictions = log_reg.predictions(model, X_test, threshold)

confusion_matrix = log_reg.generate_confusion_matrix(y_test, predictions)

# This flattens the confusing matrix into 1d arrays from a 2d.
tn, fp, fn, tp = confusion_matrix.ravel()

# Displays the raw numbers for the classification results based on threshold
st.subheader("Classification Results")

col1, col2, col3, col4 = st.columns(4)

col1.metric("True Positives", tp)
col2.metric("True Negatives", tn)
col3.metric("False Positives", fp)
col4.metric("False Negatives", fn)

# Graph of the decision threshold and it's impact on the classifications
st.subheader("Decision Threshold Visualization")

st.write("The horizontal line represents the decision threshold. Moving the threshold changes which observations are classified as positive or negative.")

# Creates a smaller plot and styling it
fig, ax = plt.subplots(
    figsize=(12, 8),
    facecolor="#f2f2f2"
)

ax.set_facecolor("#f2f2f2")

ax.grid(
    True,
    color="#e5e5e5",
    linestyle="-",
    linewidth=1.5,
    zorder=0
)

# Gets the x values and staggers the y values to prevent overlaps

rng = np.random.default_rng(42)
x_values = (y_test.astype(float) + rng.normal(0, 0.035, size=len(y_test)))

# Identify TP / TN / FP / FN

true_positives = ((y_test == 1) & (predictions == 1))

false_positives = ((y_test == 0) & (predictions == 1))

true_negatives = ((y_test == 0) & (predictions == 0))

false_negatives = ((y_test == 1) & (predictions == 0))


# Plot true positives in Q1
ax.scatter(
    x_values[true_positives],
    probabilities[true_positives],
    color="blue",
    s=55,
    alpha=0.85,
    label="True Positive",
    zorder=4
)

# False Positives in Q2
ax.scatter(
    x_values[false_positives],
    probabilities[false_positives],
    color="orange",
    s=55,
    alpha=0.85,
    label="False Positive",
    zorder=4
)


# True negatives in Q3
ax.scatter(
    x_values[true_negatives],
    probabilities[true_negatives],
    color="green",
    s=55,
    alpha=0.85,
    label="True Negative",
    zorder=4
)

# False Negatives in Q4
ax.scatter(
    x_values[false_negatives],
    probabilities[false_negatives],
    color="red",
    s=55,
    alpha=0.85,
    label="False Negative",
    zorder=4
)

# Style the decision boundary line and the vertical line
ax.axvline(
    0.5,
    color="black",
    linestyle="--",
    linewidth=2,
    zorder=2
)

ax.axhline(
    threshold,
    color="black",
    linestyle="--",
    linewidth=2.5,
    zorder=2
)


# Quadrant labels
ax.text(
    0.25,
    threshold + (1 - threshold) / 2,
    "Q2\nFALSE POSITIVES",
    ha="center",
    va="center",
    fontsize=15,
    fontweight="bold",
    alpha=0.75
)

ax.text(
    0.75,
    threshold + (1 - threshold) / 2,
    "Q1\nTRUE POSITIVES",
    ha="center",
    va="center",
    fontsize=15,
    fontweight="bold",
    alpha=0.75
)

ax.text(
    0.25,
    threshold / 2,
    "Q3\nTRUE NEGATIVES",
    ha="center",
    va="center",
    fontsize=15,
    fontweight="bold",
    alpha=0.75
)

ax.text(
    0.75,
    threshold / 2,
    "Q4\nFALSE NEGATIVES",
    ha="center",
    va="center",
    fontsize=15,
    fontweight="bold",
    alpha=0.75
)

# Axis formatting
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.05, 1.05)
ax.set_xticks([0, 1])

ax.set_xticklabels(
    [
        "Actual Negative (0)",
        "Actual Positive (1)"
    ],
    fontsize=12
)

ax.set_yticks([0, threshold, 1])

ax.set_yticklabels(["0", f"Threshold = {threshold:.2f}", "1"], fontsize=12)

# Labels:

ax.set_xlabel("Actual Class", fontsize=14, labelpad=10)
ax.set_ylabel("Predicted Probability", fontsize=14, labelpad=10)

# Title

ax.set_title(f"Logistic Regression Classification at Threshold {threshold:.2f}",
    fontsize=18,
    fontweight="bold",
    pad=15
)

# Legend
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.08),
    ncol=4,
    fontsize=11
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
st.pyplot(fig)

st.info(
    """
    There isn't a sigmoid on the graph because this model uses all 30 features to make a prediction. Therefore, 
    no one feature is able to be plotted against probability to show a meaningful sigmoid curve.
    """
)