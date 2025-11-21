"""
AI Clustering & Insight Dashboard
Single-file Streamlit app implementing:
- data upload / sample dataset
- preprocessing (scaling, imputation, encoding)
- dimensionality reduction (PCA, t-SNE, UMAP)
- clustering (KMeans, DBSCAN, GMM, OPTICS, KMedoids-if-available)
- cluster scoring (silhouette, Davies-Bouldin, Calinski-Harabasz)
- cluster profiling (feature means, top features, textual summary)
- interactive Plotly visualizations and tables

Usage:
    pip install streamlit scikit-learn pandas numpy plotly umap-learn hdbscan scikit-learn-extra
    streamlit run ai_clustering_dashboard.py

Notes:
- Some libraries (umap, scikit-learn-extra) are optional; the app falls back gracefully.
- To integrate GPT summarization: add your API call where indicated.

"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

# Optional imports
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

try:
    from sklearn_extra.cluster import KMedoids
    KMEDOIDS_AVAILABLE = True
except Exception:
    KMEDOIDS_AVAILABLE = False

# Optional Groq import will be attempted at runtime when needed
try:
    from groq import Groq
    GROQ_PY_AVAILABLE = True
except Exception:
    GROQ_PY_AVAILABLE = False

st.set_page_config(layout="wide", page_title="AI Clustering & Insight Dashboard")

# ------------------------- Helpers -------------------------
@st.cache_data
def load_sample_data(name="iris"):
    if name == "iris":
        from sklearn.datasets import load_iris
        X = load_iris(as_frame=True)
        df = X.frame
        return df
    elif name == "wine":
        from sklearn.datasets import load_wine
        X = load_wine(as_frame=True)
        df = X.frame
        return df
    else:
        return pd.DataFrame()


def preprocess(df, numeric_features=None, categorical_features=None, drop_na_threshold=0.5):
    # Basic automatic detection
    df = df.copy()
    # Drop columns with too many missing values
    df = df.loc[:, df.isnull().mean() <= (1 - drop_na_threshold)]

    if numeric_features is None:
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if categorical_features is None:
        categorical_features = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # Impute numerical
    imputer_num = SimpleImputer(strategy='median')
    X_num = pd.DataFrame(imputer_num.fit_transform(df[numeric_features]), columns=numeric_features)

    # Encode categorical (one-hot, limited cardinality)
    X_cat = pd.DataFrame()
    if len(categorical_features) > 0:
        for c in categorical_features:
            if df[c].nunique() <= 20:
                d = pd.get_dummies(df[c].astype(str), prefix=c)
                X_cat = pd.concat([X_cat, d], axis=1)
            else:
                # high cardinality -> frequency encoding
                freq = df[c].value_counts(normalize=True)
                X_cat[c + '_freq'] = df[c].map(freq).fillna(0)

    X = pd.concat([X_num, X_cat], axis=1)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_scaled, numeric_features, categorical_features


def reduce_dim(X, method='PCA', n_components=2, random_state=42, perplexity=30):
    if method == 'PCA':
        pca = PCA(n_components=n_components, random_state=random_state)
        proj = pca.fit_transform(X)
        return proj
    elif method == 't-SNE':
        tsne = TSNE(n_components=n_components, random_state=random_state, perplexity=perplexity)
        proj = tsne.fit_transform(X)
        return proj
    elif method == 'UMAP' and UMAP_AVAILABLE:
        reducer = umap.UMAP(n_components=n_components, random_state=random_state)
        proj = reducer.fit_transform(X)
        return proj
    else:
        # fallback to PCA
        pca = PCA(n_components=n_components, random_state=random_state)
        proj = pca.fit_transform(X)
        return proj


def run_clustering(X, alg='KMeans', **kwargs):
    if alg == 'KMeans':
        k = kwargs.get('n_clusters', 3)
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
    elif alg == 'DBSCAN':
        eps = kwargs.get('eps', 0.5)
        min_samples = kwargs.get('min_samples', 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
    elif alg == 'GMM':
        k = kwargs.get('n_clusters', 3)
        model = GaussianMixture(n_components=k, random_state=42)
        labels = model.fit_predict(X)
    elif alg == 'OPTICS':
        min_samples = kwargs.get('min_samples', 5)
        model = OPTICS(min_samples=min_samples)
        labels = model.fit_predict(X)
    elif alg == 'KMedoids' and KMEDOIDS_AVAILABLE:
        k = kwargs.get('n_clusters', 3)
        model = KMedoids(n_clusters=k, random_state=42, method='pam')
        labels = model.fit_predict(X)
    else:
        # fallback to KMeans
        k = kwargs.get('n_clusters', 3)
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
    return labels


def cluster_scores(X, labels):
    scores = {}
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels != -1])  # ignore noise label for DBSCAN
    try:
        if n_clusters > 1:
            scores['silhouette'] = float(silhouette_score(X, labels))
        else:
            scores['silhouette'] = None
    except Exception:
        scores['silhouette'] = None
    try:
        scores['davies_bouldin'] = float(davies_bouldin_score(X, labels)) if len(set(labels)) > 1 else None
    except Exception:
        scores['davies_bouldin'] = None
    try:
        scores['calinski_harabasz'] = float(calinski_harabasz_score(X, labels)) if len(set(labels)) > 1 else None
    except Exception:
        scores['calinski_harabasz'] = None
    scores['n_clusters'] = int(n_clusters)
    # cluster sizes
    vals, counts = np.unique(labels, return_counts=True)
    sizes = dict(zip([int(v) for v in vals], [int(c) for c in counts]))
    scores['sizes'] = sizes
    return scores


def profile_clusters(original_df, X_scaled, labels, numeric_features, top_n_features=5):
    df = original_df.reset_index(drop=True).copy()
    df['_cluster'] = labels
    profiles = {}
    for c in sorted(df['_cluster'].unique()):
        sub = df[df['_cluster'] == c]
        prof = {}
        prof['size'] = len(sub)
        # numeric feature means
        if len(numeric_features) > 0:
            means = sub[numeric_features].mean().sort_values(ascending=False)
            prof['top_numeric_means'] = means.head(top_n_features).to_dict()
        else:
            prof['top_numeric_means'] = {}
        # top categories frequency
        cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        cat_summary = {}
        for cc in cat_cols:
            cat_summary[cc] = sub[cc].value_counts(normalize=True).head(3).to_dict()
        prof['top_categorical'] = cat_summary
        # simple textual summary
        prof['summary'] = generate_text_summary(c, prof)
        profiles[int(c)] = prof
    return profiles


def generate_text_summary(cluster_label, prof):
    # Simple template-based summary. Replace or extend with GPT API if desired.
    s = f"Cluster {cluster_label}: {prof['size']} items."
    if prof['top_numeric_means']:
        nums = ', '.join([f"{k}: {v:.2f}" for k, v in prof['top_numeric_means'].items()])
        s += " Top numeric features (mean): " + nums + '.'
    if prof['top_categorical']:
        for col, vals in prof['top_categorical'].items():
            if vals:
                top = ', '.join([f"{k} ({v*100:.0f}%)" for k, v in vals.items()])
                s += f" {col}-> {top}."
    return s


def resolve_api_key(provider, openai_input=None, groq_input=None):
    """Resolve API key from environment, Streamlit secrets, or provided sidebar input.

    Returns the key (string) or None.
    """
    key = None
    if provider == 'OpenAI':
        key = os.getenv('OPENAI_API_KEY')
        if not key:
            try:
                key = st.secrets.get('OPENAI_API_KEY')
            except Exception:
                pass
        if not key and openai_input:
            key = openai_input
    elif provider == 'Groq':
        key = os.getenv('GROQ_API_KEY')
        if not key:
            try:
                key = st.secrets.get('GROQ_API_KEY')
            except Exception:
                pass
        if not key and groq_input:
            key = groq_input
    return key


def generate_groq_summary(profiles, groq_key):
    """Generate a natural-language summary using Groq chat completions.

    `profiles` is a dict mapping cluster -> profile dict (as produced by `profile_clusters`).
    Returns the generated text or raises an exception.
    """
    if not GROQ_PY_AVAILABLE:
        raise RuntimeError('Le package `groq` n\'est pas installé dans l\'environnement.')

    # Build a concise prompt including profiles
    prompt_lines = ["Tu es un assistant qui résume des profils de clusters. Donne un résumé synthétique par cluster."]
    for c, prof in sorted(profiles.items()):
        prompt_lines.append(f"Cluster {c} (taille {prof['size']}):")
        if prof.get('top_numeric_means'):
            nums = '; '.join([f"{k}={v:.2f}" for k, v in prof['top_numeric_means'].items()])
            prompt_lines.append(f"  Numeriques: {nums}")
        if prof.get('top_categorical'):
            cats = []
            for col, vals in prof['top_categorical'].items():
                if vals:
                    cats.append(f"{col}: " + ','.join([f"{k}({v*100:.0f}%)" for k, v in vals.items()]))
            if cats:
                prompt_lines.append("  Categoriels: " + ' | '.join(cats))
        if prof.get('summary'):
            prompt_lines.append(f"  Résumé template: {prof['summary']}")

    prompt = "\n".join(prompt_lines)

    client = Groq(api_key=groq_key)
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile"
    )
    # extract content
    content = response.choices[0].message.content
    return content

# ------------------------- UI -------------------------
st.title("AI Clustering & Insight Dashboard")

with st.sidebar:
    st.header("Data")
    data_source = st.radio("Charger des données:", ("Sample: iris", "Sample: wine", "Upload CSV"))
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload your CSV file", type=['csv', 'xlsx'])
    else:
        uploaded_file = None

    st.markdown("---")
    st.header("Reduction & Clustering")
    dr_method = st.selectbox("Technique de réduction", options=["PCA", "t-SNE"] + (["UMAP"] if UMAP_AVAILABLE else []))
    dr_components = st.selectbox("Dimensions de projection", options=[2, 3], index=0)
    clustering_alg = st.selectbox("Algorithme de clustering", options=["KMeans", "DBSCAN", "GMM", "OPTICS"] + (["KMedoids"] if KMEDOIDS_AVAILABLE else []))

    # hyperparameters
    st.subheader("Hyperparamètres")
    if clustering_alg in ['KMeans', 'GMM', 'KMedoids']:
        n_clusters = st.slider("n_clusters (k)", min_value=2, max_value=20, value=3)
    else:
        n_clusters = None
    eps = st.slider("eps (DBSCAN)", min_value=0.01, max_value=5.0, value=0.5, step=0.01)
    min_samples = st.slider("min_samples (DBSCAN/OPTICS)", min_value=1, max_value=50, value=5)

    st.markdown("---")
    st.header("Profiling & Export")
    gen_text_summary = st.checkbox("Générer résumé textuel automatique (template)", value=True)

    st.markdown("\n")
    st.button("Run Analysis", key='run')

    # GPT / LLM configuration (Groq is the default provider)
    st.markdown("---")
    st.header("LLM Summaries (Groq)")
    enable_gpt = st.checkbox("Activer résumé via LLM (Groq) (clé requise)", value=False)
    if enable_gpt:
        groq_key_input = st.text_input("Groq API Key (gsk-...)", type="password")
    else:
        groq_key_input = None

# Load data
if data_source == "Sample: iris":
    df = load_sample_data('iris')
elif data_source == "Sample: wine":
    df = load_sample_data('wine')
elif data_source == "Upload CSV" and uploaded_file is not None:
    try:
        if str(uploaded_file.name).endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Impossible de lire le fichier: {e}")
        st.stop()
else:
    st.info("Chargez des données échantillons ou importez un CSV pour commencer.")
    st.stop()

st.write("### Aperçu des données")
st.dataframe(df.head())

# Automatic feature detection and preprocessing
X_scaled, numeric_features, categorical_features = preprocess(df)

st.write(f"Features détectées: {len(X_scaled.columns)} (numériques: {len(numeric_features)}, catégorielles encodées: {len(categorical_features)})")

# DR
proj = reduce_dim(X_scaled, method=dr_method, n_components=dr_components)
proj_df = pd.DataFrame(proj, columns=[f"dim_{i+1}" for i in range(proj.shape[1])])

# Run clustering
cluster_kwargs = {}
if n_clusters is not None:
    cluster_kwargs['n_clusters'] = n_clusters
cluster_kwargs['eps'] = eps
cluster_kwargs['min_samples'] = min_samples
labels = run_clustering(X_scaled.values, alg=clustering_alg, **cluster_kwargs)

# Scores & profiles
scores = cluster_scores(X_scaled.values, labels)
profiles = profile_clusters(df, X_scaled, labels, numeric_features)

# Visualization layout
col1, col2 = st.columns((2, 1))

with col1:
    st.subheader("Projection & Clusters")
    if dr_components == 2:
        fig = px.scatter(proj_df, x='dim_1', y='dim_2', color=labels.astype(str), hover_data=[df.index], title='Projection 2D des clusters')
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.scatter_3d(proj_df, x='dim_1', y='dim_2', z='dim_3', color=labels.astype(str), hover_data=[df.index], title='Projection 3D des clusters')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Comparaison des indices de clustering")
    metrics_df = pd.DataFrame({
        'metric': ['silhouette', 'davies_bouldin', 'calinski_harabasz'],
        'value': [scores.get('silhouette'), scores.get('davies_bouldin'), scores.get('calinski_harabasz')]
    })
    st.table(metrics_df)

with col2:
    st.subheader("Profil des clusters")
    st.write(f"Nombre de clusters détectés: {scores['n_clusters']}")
    st.write("Taille par label (inclut -1 pour bruit si présent):")
    st.write(scores['sizes'])

    # show profiles
    for c, prof in profiles.items():
        with st.expander(f"Cluster {c} — {prof['size']} items"):
            st.write(prof['summary'])
            st.write("Top numeric means:")
            st.write(prof['top_numeric_means'])
            st.write("Top categorical distributions:")
            st.write(prof['top_categorical'])
            
    # LLM summary via Groq (default provider)
    if enable_gpt:
        if not GROQ_PY_AVAILABLE:
            st.error('Le package `groq` n\'est pas installé. Installez-le (`pip install groq`) pour utiliser Groq.')
        else:
            resolved = resolve_api_key('Groq', groq_input=groq_key_input)
            if not resolved:
                st.error('Clé Groq introuvable. Définissez-la dans la barre latérale ou via la variable d\'environnement `GROQ_API_KEY`.')
            else:
                if groq_key_input:
                    os.environ['GROQ_API_KEY'] = groq_key_input

                llm_prompt_lines = ["Voici les profils de clusters détectés :"]
                for cid, prof in sorted(profiles.items()):
                    llm_prompt_lines.append(f"Cluster {cid} — taille: {prof['size']}")
                    if prof.get('top_numeric_means'):
                        nums = '; '.join([f"{k}={v:.2f}" for k, v in prof['top_numeric_means'].items()])
                        llm_prompt_lines.append(f"  Numeriques: {nums}")
                    if prof.get('top_categorical'):
                        cats = []
                        for col, vals in prof['top_categorical'].items():
                            if vals:
                                cats.append(f"{col}: " + ','.join([f"{k}({v*100:.0f}%)" for k, v in vals.items()]))
                        if cats:
                            llm_prompt_lines.append(f"  Categoriques: {' | '.join(cats)}")
                    if prof.get('summary'):
                        llm_prompt_lines.append(f"  Resume template: {prof['summary']}")

                llm_prompt = "\n".join(llm_prompt_lines)

                try:
                    client = Groq(api_key=os.environ["GROQ_API_KEY"])
                    response = client.chat.completions.create(
                        messages=[{"role": "user", "content": llm_prompt}],
                        model="llama-3.3-70b-versatile"
                    )
                    llm_output = response.choices[0].message.content
                    st.subheader("Résumé LLM (Groq)")
                    st.write(llm_output)

                    st.download_button(
                        label="Télécharger le rapport LLM",
                        data=llm_output,
                        file_name="rapport_llm.txt",
                        mime="text/plain"
                    )

                except Exception as e:
                    st.error(f"Erreur lors de l'appel au LLM : {e}")


# Radar chart for a selected cluster
st.markdown("---")
st.subheader("Radar chart: comparer clusters sur variables numériques")
sel_vars = st.multiselect("Choisir variables numériques pour radar (au moins 3)", options=numeric_features, default=numeric_features[:5])
if len(sel_vars) >= 3:
    radar_df = df.copy()
    radar_df['_cluster'] = labels
    agg = radar_df.groupby('_cluster')[sel_vars].mean().reset_index()
    # normalize for radar (0-1)
    norm = (agg[sel_vars] - agg[sel_vars].min()) / (agg[sel_vars].max() - agg[sel_vars].min() + 1e-9)
    categories = sel_vars
    fig = go.Figure()
    for _, row in pd.concat([agg[['_cluster']], norm], axis=1).iterrows():
        fig.add_trace(go.Scatterpolar(r=row[categories].values.tolist(), theta=categories, fill='toself', name=f"Cluster {int(row['_cluster'])}"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Choisissez au moins 3 variables numériques pour afficher le radar chart.")

# Heatmap of cluster-feature means
st.subheader("Heatmap: moyennes par cluster")
heat_df = df.copy()
heat_df['_cluster'] = labels
cluster_means = heat_df.groupby('_cluster')[numeric_features].mean()
if not cluster_means.empty:
    fig = px.imshow(cluster_means.T, labels=dict(x='cluster', y='feature', color='mean'), x=cluster_means.index.astype(str))
    st.plotly_chart(fig, use_container_width=True)

# Export results
st.markdown("---")
st.subheader("Export")
export_df = df.copy()
export_df['_cluster'] = labels
csv = export_df.to_csv(index=False).encode('utf-8')
st.download_button(label="Télécharger résultats (CSV)", data=csv, file_name='clustered_results.csv', mime='text/csv')

st.success("Analyse terminée — modifiez les hyperparamètres et relancez pour comparer.")

# Next steps & tips
st.markdown("---")
st.header("Conseils & prochaines étapes")
st.markdown("- Ajouter validation croisée de stabilité (bootstrap clustering) pour mesurer la stabilité des clusters.\n- Utiliser HDBSCAN pour trouver clusters de densité robustes.\n- Intégrer un assistant GPT pour résumés textuels, ou un modèle local (distilBERT) pour la génération.\n- Ajouter un tableau comparatif multi-run (grid search) pour visualiser indices sur plusieurs combinaisons d'algos/hyperparamètres.")

st.info("Si tu veux, je peux: 1) t'aider à ajouter l'intégration GPT (avec exemple d'appel), 2) ajouter HDBSCAN et la stabilité bootstrap, 3) modulariser l'app en plusieurs fichiers.")
