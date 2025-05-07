import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def show_education_tab():
    st.header("Understanding Skewness and Transformations")

    st.markdown("""
    ### Key Terms and Definitions

    **Normal Distribution**  
    A symmetrical, bell-shaped curve where most values are near the middle.

    **Right-Skewed**  
    Most values are small, but a few very large ones stretch the graph to the right. This makes the average (mean) higher than the "typical" value.

    **Left-Skewed**  
    Most values are high, but a few very small ones drag the curve to the left.

    **Transformation**  
    A mathematical adjustment that reshapes data to reduce skew and make patterns easier to see.

    **Z-score (Standardization)**  
    Rescales data so the average is 0 and standard deviation is 1. Useful for comparing values measured on different scales.
    """)

    st.subheader("Explaining Skew with a Wallet Example")

    st.markdown("""
    Imagine asking students in a class how much money they have in their wallets:

    - Most students say between 10 and 20 dollars.
    - One student has 500 dollars!

    This one large number increases the average, even though it's not typical of most students.

    This is what right-skewed data looks like — and how we can fix it.
    """)

    money = np.array([10, 12, 13, 14, 15, 16, 18, 20, 500])

    mean_val = round(np.mean(money), 2)
    median_val = round(np.median(money), 2)

    # Apply log transformation
    transformed_money = np.log1p(money)
    trans_mean = round(np.mean(transformed_money), 2)
    trans_median = round(np.median(transformed_money), 2)

    # Create both plots in columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original Data (Skewed)**")
        fig, ax = plt.subplots()
        sns.histplot(money, bins=20, kde=True, ax=ax)
        ax.axvline(mean_val, color="red", linestyle="--", label=f"Mean = {mean_val}")
        ax.axvline(median_val, color="blue", linestyle="--", label=f"Median = {median_val}")
        ax.set_title("Original Wallet Data")
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.markdown("**Log-Transformed Data (Less Skewed)**")
        fig2, ax2 = plt.subplots()
        sns.histplot(transformed_money, bins=20, kde=True, ax=ax2)
        ax2.axvline(trans_mean, color="red", linestyle="--", label=f"Mean = {trans_mean}")
        ax2.axvline(trans_median, color="blue", linestyle="--", label=f"Median = {trans_median}")
        ax2.set_title("Log-Transformed Wallet Data")
        ax2.legend()
        st.pyplot(fig2)

    st.markdown("""
    **What this shows:**

    - The original data is skewed by one large outlier.
    - The log transformation compresses that outlier, making the data more balanced.
    - The mean and median become closer, making the average more representative.
    """)



    st.subheader("Visualizing Common Distribution Shapes")

    normal_data = np.random.normal(loc=0, scale=1, size=1000)
    right_skewed = np.random.exponential(scale=2, size=1000)
    left_skewed = -np.random.exponential(scale=2, size=1000)

    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 4))
    sns.histplot(normal_data, bins=40, kde=True, ax=axes1[0])
    axes1[0].set_title("Normal Distribution")

    sns.histplot(right_skewed, bins=40, kde=True, ax=axes1[1])
    axes1[1].set_title("Right-Skewed Distribution")

    sns.histplot(left_skewed, bins=40, kde=True, ax=axes1[2])
    axes1[2].set_title("Left-Skewed Distribution")

    st.pyplot(fig1)

    st.markdown("""
    **What to notice:**

    - Right-skewed: most values are small, with a few large ones
    - Left-skewed: most values are large, with a few small ones
    - Normal: values are balanced on both sides
    """)

    st.subheader("Transforming Right-Skewed Data")

    log_trans = np.log1p(right_skewed)
    sqrt_trans = np.sqrt(right_skewed)
    zscore_trans = StandardScaler().fit_transform(right_skewed.reshape(-1, 1)).flatten()

    fig2, axes2 = plt.subplots(1, 4, figsize=(24, 4))
    sns.histplot(right_skewed, bins=40, kde=True, ax=axes2[0])
    axes2[0].set_title("Original")

    sns.histplot(log_trans, bins=40, kde=True, ax=axes2[1])
    axes2[1].set_title("Log Transform")

    sns.histplot(sqrt_trans, bins=40, kde=True, ax=axes2[2])
    axes2[2].set_title("Square Root Transform")

    sns.histplot(zscore_trans, bins=40, kde=True, ax=axes2[3])
    axes2[3].set_title("Z-score Standardization")

    st.pyplot(fig2)

    st.markdown("""
    **When to use:**

    - Log: Best for strong right skew (such as income or large numeric ranges)
    - Square Root: Useful for moderate skew or count data
    - Z-score: Helps compare features on the same scale, but does not fix skew
    """)

    st.subheader("Skewed X-Axis in Relationships")

    x = np.random.exponential(scale=2, size=500)
    noise = np.random.normal(0, 3, size=500)
    y = 3 * x + noise

    log_x = np.log1p(x)
    sqrt_x = np.sqrt(x)

    st.markdown("Here we show a right-skewed input (X) and how it affects a relationship with Y.")

    fig3, axes3 = plt.subplots(1, 3, figsize=(21, 5))

    axes3[0].scatter(x, y, alpha=0.5)
    axes3[0].set_title("Original X")
    axes3[0].set_xlabel("X")
    axes3[0].set_ylabel("Y")

    axes3[1].scatter(log_x, y, alpha=0.5)
    axes3[1].set_title("Log-Transformed X")
    axes3[1].set_xlabel("log1p(X)")

    axes3[2].scatter(sqrt_x, y, alpha=0.5)
    axes3[2].set_title("Square Root Transformed X")
    axes3[2].set_xlabel("sqrt(X)")

    st.pyplot(fig3)

    st.markdown("""
    **What to see:**

    - Original X shows a curved or bunched-up shape
    - Transformed X creates a straighter, more useful relationship with Y
    """)

    st.subheader("Real-World Examples of When and Why to Transform")

    st.markdown("""
    **Transforming Y (the value you're predicting)**

    - Home prices: A few very expensive homes skew the data
    - Hospital stay length: Some people stay far longer than average
    - Number of purchases: Most customers buy 1–2 items, a few buy 100+

    These are right-skewed outcomes where transformation helps reduce outlier impact and improve prediction.

    **Transforming X (the input variable)**

    - Income: A few people earn far more than the rest
    - Population: A few cities are much larger than most
    - App usage: Some users sign up years ago while most are new

    These are right-skewed inputs that can make relationships harder to see unless transformed.

    **Transforming both X and Y**

    - Income vs. spending: Both may be skewed in retail or finance
    - Study time vs. test score: Some students study more and get much higher scores

    **When not to transform**

    - If the data is already close to normal
    - If your model doesn’t require it (like tree-based models)
    - If interpretability in original units is more important than model performance

    """)


def show_skew_explorer_tab():
    st.header("Explore Skewness")

    st.markdown("""
    Use the slider below to control how skewed the X data is.  
    Higher values create stronger right-skew. Y is generated using `Y = 3X + noise`.

    This lets you *see* how skew affects distributions and relationships.
    """)

    skew_strength = st.slider("Choose skew strength (scale of exponential distribution):", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

    show_table = st.checkbox("Show raw X and Y values")

    # Generate data
    x = np.random.exponential(scale=skew_strength, size=300)
    noise = np.random.normal(0, 2, size=300)
    y = 3 * x + noise

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Histogram of X")
        fig_x, ax_x = plt.subplots()
        sns.histplot(x, bins=40, kde=True, ax=ax_x)
        ax_x.set_title(f"X Distribution (Scale = {skew_strength})")
        st.pyplot(fig_x)

    with col2:
        st.subheader("Scatter Plot (X vs Y)")
        fig_xy, ax_xy = plt.subplots()
        ax_xy.scatter(x, y, alpha=0.5)
        ax_xy.set_xlabel("X")
        ax_xy.set_ylabel("Y")
        ax_xy.set_title("Y = 3X + noise")
        st.pyplot(fig_xy)

    if show_table:
        st.subheader("Raw Data (First 30 Rows)")
        st.dataframe({"X": x[:30], "Y": y[:30]})
