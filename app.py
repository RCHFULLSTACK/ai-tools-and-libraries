import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import time
import re
from collections import Counter
import base64
import json

# Konfigurera sidan
st.set_page_config(page_title="AI-bibliotek Tracker", page_icon="🧠", layout="wide")

# Titelsektionen
st.title("🧠 AI-verktyg & Bibliotek Analys")
st.markdown("Utforska vilka AI-bibliotek som är mest populära på GitHub")

# GitHub API-funktioner
def get_top_ai_repositories(per_page=30):
    """Hämta topp AI-repositories"""
    url = f"https://api.github.com/search/repositories?q=artificial+intelligence+OR+machine+learning+OR+deep+learning&sort=stars&order=desc&per_page={per_page}"
    response = requests.get(url)
    return response.json()

def extract_languages_data():
    """Hämta data om populära AI-relaterade språk"""
    languages = ["Python", "JavaScript", "Java", "C++", "TypeScript", "R", "Julia", "Go", "Rust", "C#"]
    language_data = []
    
    with st.spinner("Hämtar språkdata från GitHub..."):
        for language in languages:
            url = f"https://api.github.com/search/repositories?q=artificial+intelligence+OR+machine+learning+language:{language}&sort=stars&order=desc&per_page=10"
            response = requests.get(url)
            data = response.json()
            if 'items' in data:
                total_stars = sum(repo['stargazers_count'] for repo in data['items'])
                avg_stars = total_stars / len(data['items']) if len(data['items']) > 0 else 0
                language_data.append({
                    'language': language,
                    'repositories': len(data['items']),
                    'stars': total_stars,
                    'avg_stars': avg_stars
                })
            time.sleep(0.7)  # Pausa för att respektera API-gränser
    
    return pd.DataFrame(language_data)

def get_popular_ai_libraries():
    """Hämta populära AI-bibliotek baserat på fördefinierad data"""
    ai_libraries = [
        {"name": "TensorFlow", "category": "Deep Learning", "language": "Python", "stars": 172000, "description": "End-to-end ML platform"},
        {"name": "PyTorch", "category": "Deep Learning", "language": "Python", "stars": 64500, "description": "Tensors and dynamic neural networks"},
        {"name": "scikit-learn", "category": "Machine Learning", "language": "Python", "stars": 53800, "description": "ML algorithms and tools"},
        {"name": "Keras", "category": "Deep Learning", "language": "Python", "stars": 58200, "description": "Deep learning API"},
        {"name": "Transformers", "category": "NLP", "language": "Python", "stars": 45700, "description": "State-of-the-art NLP"},
        {"name": "spaCy", "category": "NLP", "language": "Python", "stars": 25300, "description": "Industrial-strength NLP"},
        {"name": "NLTK", "category": "NLP", "language": "Python", "stars": 12100, "description": "Natural language toolkit"},
        {"name": "XGBoost", "category": "Machine Learning", "language": "C++", "stars": 24600, "description": "Gradient boosting framework"},
        {"name": "LightGBM", "category": "Machine Learning", "language": "C++", "stars": 15200, "description": "Gradient boosting framework"},
        {"name": "TensorFlow.js", "category": "Deep Learning", "language": "JavaScript", "stars": 17800, "description": "JavaScript ML library"},
        {"name": "fastai", "category": "Deep Learning", "language": "Python", "stars": 24100, "description": "Deep learning library"},
        {"name": "Gensim", "category": "NLP", "language": "Python", "stars": 14300, "description": "Topic modeling and embeddings"},
        {"name": "JAX", "category": "Deep Learning", "language": "Python", "stars": 19600, "description": "High-performance ML research"},
        {"name": "CatBoost", "category": "Machine Learning", "language": "C++", "stars": 7200, "description": "Gradient boosting library"},
        {"name": "ML.NET", "category": "Machine Learning", "language": "C#", "stars": 8400, "description": ".NET ML framework"},
        {"name": "TorchAudio", "category": "Speech", "language": "Python", "stars": 3700, "description": "Audio processing tools"},
        {"name": "TorchVision", "category": "Computer Vision", "language": "Python", "stars": 14500, "description": "Computer vision datasets and models"},
        {"name": "OpenCV", "category": "Computer Vision", "language": "C++", "stars": 71200, "description": "Computer vision library"},
        {"name": "Langchain", "category": "LLMs", "language": "Python", "stars": 52000, "description": "Building applications with LLMs"},
        {"name": "Ray", "category": "Distributed Computing", "language": "Python", "stars": 28500, "description": "Distributed ML framework"}
    ]
    return pd.DataFrame(ai_libraries)

def get_library_frameworks():
    """Hämta data om populära AI-frameworks och deras språkanvändning"""
    library_frameworks = [
        {"library": "TensorFlow", "language": "Python", "count": 42},
        {"library": "TensorFlow", "language": "JavaScript", "count": 14},
        {"library": "TensorFlow", "language": "C++", "count": 8},
        {"library": "TensorFlow", "language": "Java", "count": 5},
        {"library": "PyTorch", "language": "Python", "count": 47},
        {"library": "PyTorch", "language": "C++", "count": 12},
        {"library": "PyTorch", "language": "JavaScript", "count": 4},
        {"library": "scikit-learn", "language": "Python", "count": 38},
        {"library": "Keras", "language": "Python", "count": 32},
        {"library": "Keras", "language": "R", "count": 5},
        {"library": "Transformers", "language": "Python", "count": 40},
        {"library": "spaCy", "language": "Python", "count": 28},
        {"library": "NLTK", "language": "Python", "count": 25},
        {"library": "XGBoost", "language": "Python", "count": 30},
        {"library": "XGBoost", "language": "R", "count": 15},
        {"library": "XGBoost", "language": "Java", "count": 4},
        {"library": "LightGBM", "language": "Python", "count": 22},
        {"library": "LightGBM", "language": "R", "count": 13},
        {"library": "TensorFlow.js", "language": "JavaScript", "count": 35},
        {"library": "fastai", "language": "Python", "count": 24},
        {"library": "OpenCV", "language": "Python", "count": 28},
        {"library": "OpenCV", "language": "C++", "count": 25},
        {"library": "OpenCV", "language": "Java", "count": 12},
        {"library": "Langchain", "language": "Python", "count": 36},
        {"library": "Langchain", "language": "TypeScript", "count": 8}
    ]
    return pd.DataFrame(library_frameworks)

# Caching av data för att minska API-anrop
@st.cache_data(ttl=3600)
def load_language_data():
    return extract_languages_data()

@st.cache_data(ttl=3600)
def load_ai_libraries():
    return get_popular_ai_libraries()

@st.cache_data(ttl=3600)
def load_library_frameworks():
    return get_library_frameworks()

# Ladda data
try:
    # Ladda språkdata
    language_data = load_language_data()
    
    # Ladda biblioteksdata
    libraries = load_ai_libraries()
    
    # Ladda framework-språk-data
    library_frameworks = load_library_frameworks()
    
    # Visa statistik
    st.subheader("AI-bibliotek Översikt")
    
    # Skapa statistikrutor
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Populäraste AI-språket", language_data.iloc[0]['language'] if not language_data.empty else "N/A")
    with col2:
        st.metric("Antal AI-bibliotek", len(libraries))
    with col3:
        top_lib = libraries.sort_values('stars', ascending=False).iloc[0]['name'] if not libraries.empty else "N/A"
        st.metric("Populäraste biblioteket", top_lib)
    
    # Skapa två kolumner för visualiseringar
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Populäraste AI-biblioteken")
        fig, ax = plt.subplots(figsize=(10, 8))
        top_libs = libraries.sort_values('stars', ascending=False).head(15)
        bars = ax.barh(top_libs['name'], top_libs['stars'], color='skyblue')
        plt.xlabel('Antal stjärnor')
        plt.ylabel('Bibliotek')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("Programmeringsspråk i AI-utveckling")
        fig, ax = plt.subplots(figsize=(10, 8))
        langs_df = language_data.sort_values('stars', ascending=False)
        bars = ax.barh(langs_df['language'], langs_df['stars'], color='lightgreen')
        plt.xlabel('Antal stjärnor (totalt)')
        plt.ylabel('Språk')
        plt.tight_layout()
        st.pyplot(fig)
    
    # Visa bibliotek efter kategori
    st.subheader("AI-bibliotek efter kategori")
    categories = sorted(libraries['category'].unique())
    selected_category = st.selectbox("Välj kategori", categories)
    
    if selected_category:
        category_libs = libraries[libraries['category'] == selected_category].sort_values('stars', ascending=False)
        if not category_libs.empty:
            st.dataframe(category_libs[['name', 'language', 'stars', 'description']])
    
    # Visa detaljerad biblioteksdata
    st.subheader("Språkanvändning per AI-bibliotek")
    library_options = sorted(library_frameworks['library'].unique())
    library_to_show = st.selectbox("Välj bibliotek", library_options)
    
    if library_to_show:
        lib_langs = library_frameworks[library_frameworks['library'] == library_to_show]
        if not lib_langs.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            lib_langs = lib_langs.sort_values('count', ascending=False)
            bars = ax.barh(lib_langs['language'], lib_langs['count'], color='coral')
            plt.xlabel('Användningsfrekvens')
            plt.ylabel('Språk')
            plt.title(f'Språkanvändning för {library_to_show}')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Hitta biblioteksinformation
            lib_info = libraries[libraries['name'] == library_to_show].iloc[0] if not libraries[libraries['name'] == library_to_show].empty else None
            if lib_info is not None:
                st.markdown(f"**Kategori:** {lib_info['category']}")
                st.markdown(f"**Primärt språk:** {lib_info['language']}")
                st.markdown(f"**Beskrivning:** {lib_info['description']}")
        else:
            st.info(f"Ingen språkdata tillgänglig för {library_to_show}")
    
except Exception as e:
    st.error(f"Ett fel uppstod: {e}")
    st.info("GitHub API har begränsningar för antal anrop. Vänta en stund eller använd API-nycklar för högre kvot.")

# Footer
st.markdown("---")
st.markdown("Data baserad på GitHub API och trender inom AI-utveckling. Uppdateras varje timme.")