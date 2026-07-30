import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="MovieIQ",
    page_icon="🎬",
    layout="wide"
)

# Load Dataset
@st.cache_data
def load_data():
    return pd.read_csv("data/clean_movies.csv")

df = load_data()

# ---------- SIDEBAR ----------
st.sidebar.title("🎛 Filters")

# Genre Filter
if "genres" in df.columns:
    genres = sorted(df["genres"].dropna().unique())
    selected_genre = st.sidebar.selectbox(
        "Select Genre",
        ["All"] + genres
    )

    if selected_genre != "All":
        df = df[df["genres"] == selected_genre]

# Vote Average Filter
min_rating = st.sidebar.slider(
    "Minimum Rating",
    float(df["vote_average"].min()),
    float(df["vote_average"].max()),
    float(df["vote_average"].min())
)

df = df[df["vote_average"] >= min_rating]

# Runtime Filter
runtime = st.sidebar.slider(
    "Runtime (minutes)",
    int(df["runtime"].min()),
    int(df["runtime"].max()),
    (
        int(df["runtime"].min()),
        int(df["runtime"].max())
    )
)

df = df[
    (df["runtime"] >= runtime[0]) &
    (df["runtime"] <= runtime[1])
]

# Popularity Filter
popularity = st.sidebar.slider(
    "Popularity",
    float(df["popularity"].min()),
    float(df["popularity"].max()),
    (
        float(df["popularity"].min()),
        float(df["popularity"].max())
    )
)

df = df[
    (df["popularity"] >= popularity[0]) &
    (df["popularity"] <= popularity[1])
]

# ---------- MAIN ----------
st.title("🎬 MovieIQ Dashboard")

total_movies = len(df)
avg_rating = df["vote_average"].mean()
avg_revenue = df["revenue"].mean()
success_rate = df["success"].mean() * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric("🎥 Total Movies", total_movies)
col2.metric("⭐ Avg Rating", f"{avg_rating:.2f}")
col3.metric("💰 Avg Revenue", f"${avg_revenue:,.0f}")
col4.metric("✅ Success Rate", f"{success_rate:.1f}%")

st.divider()

import plotly.express as px

st.header("📊 Exploratory Data Analysis")

col1, col2 = st.columns(2)

# Budget vs Revenue
with col1:
    fig1 = px.scatter(
        df,
        x="budget",
        y="revenue",
        color="success",
        hover_data=["vote_average", "runtime"],
        title="Budget vs Revenue"
    )
    st.plotly_chart(fig1, use_container_width=True)

# Success Distribution
with col2:
    fig2 = px.pie(
        df,
        names="success",
        title="Success Distribution"
    )
    st.plotly_chart(fig2, use_container_width=True)
col3, col4 = st.columns(2)

with col3:
    fig3 = px.histogram(
        df,
        x="vote_average",
        nbins=20,
        title="Vote Average Distribution"
    )
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    fig4 = px.histogram(
        df,
        x="popularity",
        nbins=30,
        title="Popularity Distribution"
    )
    st.plotly_chart(fig4, use_container_width=True)

st.subheader("🔥 Correlation Heatmap")

corr = df[
    [
        "budget",
        "revenue",
        "runtime",
        "popularity",
        "vote_average"
    ]
].corr()

fig5 = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="Viridis"
)

st.plotly_chart(fig5, use_container_width=True)

st.divider()
st.subheader("📄 Dataset Preview")

st.dataframe(df, use_container_width=True)

from scipy.stats import ttest_ind, chi2_contingency

st.divider()
st.header("🧪 Statistical Test Results")

# ------------------ T-Test ------------------

success_movies = df[df["success"] == 1]["popularity"]
failure_movies = df[df["success"] == 0]["popularity"]

t_stat, p_value = ttest_ind(success_movies, failure_movies, equal_var=False)

col1, col2 = st.columns(2)

with col1:
    st.subheader("T-Test")

    st.metric("T Statistic", f"{t_stat:.3f}")
    st.metric("P-Value", f"{p_value:.5f}")

    if p_value < 0.05:
        st.success("Popularity differs significantly between successful and unsuccessful movies.")
    else:
        st.info("No significant difference in popularity was found.")

# ------------------ Chi-Square Test ------------------

with col2:
    st.subheader("Chi-Square Test")

    if "genres" in df.columns:

        contingency = pd.crosstab(df["genres"], df["success"])

        chi2, p, dof, expected = chi2_contingency(contingency)

        st.metric("Chi-Square", f"{chi2:.3f}")
        st.metric("P-Value", f"{p:.5f}")

        if p < 0.05:
            st.success("Movie genre is significantly associated with success.")
        else:
            st.info("No significant association between genre and success.")
            import joblib
import numpy as np

st.divider()
st.header("🤖 Movie Success Prediction")

# Load Model
@st.cache_resource
def load_model():
    return joblib.load("models/random_forest.pkl")

model = load_model()

col1, col2 = st.columns(2)

with col1:
    budget = st.number_input(
        "Budget ($)",
        min_value=0.0,
        value=1000000.0
    )

    popularity = st.number_input(
        "Popularity",
        min_value=0.0,
        value=20.0
    )

with col2:
    runtime = st.number_input(
        "Runtime (minutes)",
        min_value=30,
        value=120
    )

    vote_average = st.slider(
        "Vote Average",
        0.0,
        10.0,
        7.0
    )

if st.button("Predict Success", use_container_width=True):

    features = np.array([[budget,
                          popularity,
                          runtime,
                          vote_average]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
   
    if prediction == 1:
        st.success("🎉 This movie is predicted to be Successful!")
    else:
        st.error("❌ This movie is predicted to be Unsuccessful.")

    st.subheader("Prediction Confidence")

    col1, col2 = st.columns(2)

    col1.metric(
        "Success Probability",
        f"{probability[1]*100:.2f}%"
    )

    col2.metric(
        "Failure Probability",
        f"{probability[0]*100:.2f}%"
    )

    st.progress(float(probability[1]))

st.divider()

st.header("📊 Feature Importance")

st.image(
    "assets/feature_importance.png",
    caption="Random Forest Feature Importance",
    use_container_width=True
)

st.divider()

st.header("📈 Confusion Matrix")

st.image(
    "assets/confusion_matrix.png",
    caption="Confusion Matrix",
    use_container_width=True
)


st.header("ℹ️ About MovieIQ")

st.markdown("""
### MovieIQ – Predictive Analytics on Film Success

This project predicts whether a movie will be successful using a Random Forest Classifier.

### Features
- Data Preprocessing
- Exploratory Data Analysis
- Statistical Tests
- Movie Success Prediction
- Interactive Dashboard

### Technologies
- Python
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Streamlit
""")
