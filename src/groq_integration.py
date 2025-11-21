"""
Groq LLM integration module.
Resolves API keys and generates cluster summaries via Groq.
"""

import os

# Optional Groq import
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False


def resolve_groq_api_key(sidebar_input=None):
    """
    Resolve Groq API key from multiple sources.
    Priority: sidebar input -> st.secrets -> environment variable.
    
    Args:
        sidebar_input: API key from Streamlit sidebar text_input (optional)
    
    Returns:
        key: Groq API key string, or None if not found
    """
    key = None
    
    # 1. sidebar input has priority
    if sidebar_input:
        key = sidebar_input
    else:
        # 2. environment variable
        key = os.getenv('GROQ_API_KEY')
        
        # 3. streamlit secrets (if running in st app context)
        if not key:
            try:
                import streamlit as st
                key = st.secrets.get('GROQ_API_KEY')
            except Exception:
                pass
    
    return key


def generate_groq_summary(profiles, groq_key):
    """
    Generate a natural-language cluster summary using Groq LLM.
    
    Args:
        profiles: dict of cluster profiles (from profiling.profile_clusters)
        groq_key: Groq API key
    
    Returns:
        content: LLM-generated summary text
    
    Raises:
        RuntimeError: if groq package is not installed
        Exception: on API errors
    """
    if not GROQ_AVAILABLE:
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
                prompt_lines.append("  Categoriques: " + ' | '.join(cats))
        if prof.get('summary'):
            prompt_lines.append(f"  Resume template: {prof['summary']}")

    prompt = "\n".join(prompt_lines)

    client = Groq(api_key=groq_key)
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile"
    )
    
    content = response.choices[0].message.content
    return content
