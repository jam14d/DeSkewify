import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
import requests
from io import StringIO

st.set_page_config(page_title="DeSkewify", layout="wide")
st.title("DeSkewify: Straighten the Story")

# Sidebar: Educational Guide
with st.sidebar:
    st.header("Transformation Guide")

    st.markdown("""
    **Why Use Transformations:**
    | Purpose | Description |
    |---------|-------------|
    | Normalize | Corrects skewed distributions |
    | Clarify Visuals | Makes patterns more visible |
    | Modeling Prep | Meets assumptions like normality or equal variance |

    **When to Use Each Transformation:**
    | Transformation | When to Use |
    |----------------|-------------|
    | Log10 | Exponential or right-skewed data |
    | Square Root | Moderate skew, count data |
    | Z-score | Standardize across scales |
    | Min-Max | Normalize to [0, 1] |
    | Yeo-Johnson | Works with skew, zeros, negatives |

    **Visualization Types:**
    | Type | Use Case |
    |------|----------|
    | Scatter Plot | Relationship between two continuous variables |
    | Line Plot | Trends over time or order |

    **Application Context (Continuous-Continuous):**
    | Question | Guidance |
    |----------|----------|
    | What data is suitable? | Two numeric columns |
    | What to avoid? | Categorical X unless using bar/box plots |

    **Should You Transform?**
    | Question | Answer |
    |----------|--------|
    | Should we transform Y? | Almost always a good option if it's skewed |
    | Should we transform X? | Sometimes — if it's skewed, nonlinear, or has huge scale |
    | Should we transform both? | You can, but be cautious — only do it if you can interpret both |

    ### References

    - [NIST: Skewness Explained](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35b.htm)
    - [Pandas `.skew()` Function](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.skew.html)
    - [Scikit-learn: `PowerTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html)
    - [Matplotlib](https://matplotlib.org/) 
    - [Seaborn](https://seaborn.pydata.org/)
    - [Streamlit](https://docs.streamlit.io/)
""")


# Section: App Introduction
st.markdown("""

Every dataset tells a story—but when your data is skewed, that story can get distorted.

**DeSkewify** helps you straighten the narrative by revealing the true shape of your data. 

Use the interface below to load your dataset, choose variables, and experiment with transformations and visualization options.

1. **Upload or use the example dataset**
2. **Check for skew** in numeric columns 
3. **Select variables** for X and Y axes
4. **Apply transformations** and visually compare results
5. **Review summary statistics** to understand changes

""")

# Section: Data Loading
st.header("1. Load and Preview Data")

data_source = st.radio("Choose data source:", ("Use example dataset (COVID-19)", "Upload your own CSV file"))

def load_example_data():
    url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
    response = requests.get(url)
    csv_data = StringIO(response.text)
    return pd.read_csv(csv_data)

if data_source == "Upload your own CSV file":
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Custom data uploaded successfully.")
        except Exception as e:
            st.error(f"Failed to load uploaded CSV file: {e}")
            st.stop()
    else:
        st.warning("Awaiting file upload.")
        st.stop()
else:
    df = load_example_data()
    st.info("Using example COVID-19 dataset.")

st.write("Sample data:")
st.dataframe(df.sample(5))

# Section: Column Selection

# Section: Skewness Analysis
st.header("2. Analyze Raw Data for Skewness")

num_cols = df.select_dtypes(include=np.number).columns.tolist()
if not num_cols:
    st.error("No numeric columns found in the dataset.")
    st.stop()

# Analyze skewness column-by-column
skew_data = pd.DataFrame([
    {
        "Column": col,
        "Skewness": df[col].dropna().skew(),
        "Needs Transformation": abs(df[col].dropna().skew()) > 1,
        "Suggested Transformation": (
            "Log10" if df[col].dropna().skew() > 1 else
            "Square Root" if 0.5 < df[col].dropna().skew() <= 1 else
            "None"
        )
    }
    for col in num_cols
])

st.markdown("""Columns with high skewness (>|1|) may benefit from transformation.
Below is an overview of numeric columns and whether a transformation is recommended:
""")
st.dataframe(skew_data.set_index("Column"))

st.header("2. Select Columns")

num_cols = df.select_dtypes(include=np.number).columns.tolist()
if not num_cols:
    st.error("No numeric columns found in the dataset.")
    st.stop()

x_col = st.selectbox("Select the X-axis column:", num_cols)
y_col = st.selectbox("Select the Y-axis column:", num_cols)

# Drop rows where either X or Y is missing
valid_data = df[[x_col, y_col]].dropna()
x = valid_data[x_col].values.reshape(-1, 1)
y = valid_data[y_col].values.reshape(-1, 1)

# Section: Transformation Selection
st.header("3. Apply Transformations")

x_transform = st.selectbox(
    "Transformation for X-axis:",
    ["None", "Log10", "Square Root", "Z-score", "Min-Max", "Yeo-Johnson"],
    help="Choose a mathematical transformation for the X-axis. Use Log10 or Square Root if the data is right-skewed."
)
y_transform = st.selectbox(
    "Transformation for Y-axis:",
    ["None", "Log10", "Square Root", "Z-score", "Min-Max", "Yeo-Johnson"],
    help="Choose a transformation for the Y-axis. Helpful if data is skewed or spans large scales."
)

def apply_transformation(data, method):
    try:
        if method == "Log10":
            return np.log10(data + 1)
        elif method == "Square Root":
            return np.sqrt(data)
        elif method == "Z-score":
            return StandardScaler().fit_transform(data)
        elif method == "Min-Max":
            return MinMaxScaler().fit_transform(data)
        elif method == "Yeo-Johnson":
            return PowerTransformer(method='yeo-johnson').fit_transform(data)
        else:
            return data
    except Exception as e:
        st.error(f"Transformation error: {e}")
        return data

x_trans = apply_transformation(x, x_transform)
y_trans = apply_transformation(y, y_transform)

# Section: Visualization
st.header("4. Visualization: Original vs Transformed")

plot_type = st.selectbox("Choose plot type:", ["Scatter Plot", "Line Plot"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
if plot_type == "Scatter Plot":
    ax1.scatter(x, y, alpha=0.6)
    ax2.scatter(x_trans, y_trans, alpha=0.6)
elif plot_type == "Line Plot":
    ax1.plot(x, y, alpha=0.8)
    ax2.plot(x_trans, y_trans, alpha=0.8)

ax1.set_title("Original")
ax1.set_xlabel(x_col)
ax1.set_ylabel(y_col)
ax2.set_title("Transformed")
ax2.set_xlabel(f"{x_col} ({x_transform})")
ax2.set_ylabel(f"{y_col} ({y_transform})")

st.pyplot(fig)

# Optional: Show code for transparency
import inspect

if st.checkbox("Show code used for transformations and skewness analysis"):
    code_blocks = [
        inspect.getsource(apply_transformation),
        """# Skewness calculation
skew = df[col].dropna().skew()
"""
    ]
    for block in code_blocks:
        st.code(block, language='python')



# Section: Summary Statistics
st.subheader("Summary Statistics")
st.markdown("**Y-axis Statistics:**")
y_stats = pd.DataFrame({
    "": ["Mean", "Standard Deviation", "Skewness"],
    "Original": [np.mean(y), np.std(y), pd.Series(y.flatten()).skew()],
    f"{y_transform}": [np.mean(y_trans), np.std(y_trans), pd.Series(y_trans.flatten()).skew()]
})
st.dataframe(y_stats.set_index(""))

st.markdown("**X-axis Statistics:**")
x_stats = pd.DataFrame({
    "": ["Mean", "Standard Deviation", "Skewness"],
    "Original": [np.mean(x), np.std(x), pd.Series(x.flatten()).skew()],
    f"{x_transform}": [np.mean(x_trans), np.std(x_trans), pd.Series(x_trans.flatten()).skew()]
})
st.dataframe(x_stats.set_index(""))
