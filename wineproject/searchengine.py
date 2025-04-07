import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import kagglehub

# --- Global Layout Fix ---
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        max-width: 100%;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Load dataset from KaggleHub ---
@st.cache_data
def load_data():
    path = kagglehub.dataset_download("christopheiv/winemagdata130k")
    df = pd.read_csv(f"{path}/winemag-data-130k-v2.csv", low_memory=False)
    df = df.dropna(subset=["description"])
    df = df.fillna("")
    df = df.drop_duplicates(subset=["description"]).reset_index(drop=True)
    return df

# --- Load model ---
@st.cache_resource
def load_model():
    return SentenceTransformer("fine-tuned-minilm")

model = load_model()
wine_df = load_data()

# --- Precompute embeddings ---
@st.cache_data
def get_embeddings(texts):
    return model.encode(texts, convert_to_tensor=True)

wine_texts = wine_df["description"].tolist()
wine_embeddings = get_embeddings(wine_texts)

# --- Helpers ---
def safe_get(value):
    return "" if pd.isna(value) or value is None else str(value)

def format_similarity(sim: float) -> float:
    return round(80 + (sim * 20), 2)  # Scale to 80–100

# --- Title ---
st.title("🍷 Wine Finder")

# --- Input Controls with Responsive Styling ---
st.markdown(
    """
    <style>
    .input-row {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        align-items: center;
    }
    .input-row > div {
        flex-grow: 1;
        min-width: 150px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="input-row">', unsafe_allow_html=True)

col1, col2, col3 = st.columns([10, 3, 1])
with col1:
    query = st.text_input("Describe the kind of wine you're looking for:", key="query", placeholder="e.g., a dry red wine with fruity notes and low acid")
with col2:
        sort_option = st.selectbox(
        "Sort by",
        [
            "↓ Similarity",
            "↑ Similarity",
            "↓ Price",
            "↑ Price",
            "↓ Points",
            "↑ Points",
        ],
        index=0,
        key="sort"
    )
    
with col3:
    num_results = st.number_input("Results", min_value=1, max_value=20, value=5, step=1, key="results")

st.markdown('</div>', unsafe_allow_html=True)

# --- Main Logic ---
if query:
    query_embedding = model.encode(query, convert_to_tensor=True)
    sims = util.cos_sim(query_embedding, wine_embeddings)[0]

    # Get all scores and indices, sorted by similarity
    all_scores = sims.tolist()
    all_indices = list(range(len(all_scores)))
    top_scores_and_indices = sorted(
        zip(all_scores, all_indices),
        key=lambda x: x[0],
        reverse=True
    )[:num_results]

    # Collect wine data with extra fields
    results = []
    for score, idx in top_scores_and_indices:
        wine = wine_df.iloc[idx]
        results.append({
            "wine": wine,
            "similarity": score,
            "price": wine.get("price") if wine.get("price") != "" else float("inf"),
            "points": wine.get("points") if wine.get("points") != "" else 0,
        })

    # Sort based on user choice
    reverse_sort = "↓" in sort_option
    if "Price" in sort_option:
        results.sort(key=lambda x: x["price"] if pd.notna(x["price"]) else float("inf"), reverse=reverse_sort)
    elif "Points" in sort_option:
        results.sort(key=lambda x: x["points"] if pd.notna(x["points"]) else 0, reverse=reverse_sort)
    elif "Similarity" in sort_option:
        results.sort(key=lambda x: x["similarity"], reverse=reverse_sort)

    # Display results
    st.subheader(f"Top {num_results} Matches:")

    for result in results:
        wine = result["wine"]
        score = result["similarity"]

        title = f"{safe_get(wine.get('winery'))} {safe_get(wine.get('designation'))}".strip()
        location = f"{safe_get(wine.get('region_1'))}, {safe_get(wine.get('region_2'))}".strip(", ")
        sim_score = format_similarity(score)

        st.markdown(f"### {title if title else 'Unnamed Wine'}")
        if location:
            st.write(f"📍 {location}")
        st.write(f"**Similarity Score**: {sim_score}/100")
        st.write(f"**Variety**: {wine.get('variety', 'N/A')}")
        st.write(f"**Country**: {wine.get('country', 'N/A')}")
        st.write(f"**Price**: ${wine.get('price', 'N/A')}")
        st.write(f"**Points**: {wine.get('points', 'N/A')}")
        st.write(f"**Description**: {wine.get('description')}")
        st.markdown("---")
