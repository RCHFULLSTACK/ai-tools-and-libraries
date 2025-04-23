import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import time
import re
from collections import Counter

# Konfigurera sidan
st.set_page_config(page_title="AI-bibliotek Tracker", page_icon="🧠", layout="wide")

# Titelsektionen
st.title("🧠 AI-verktyg & Bibliotek Analys")
st.markdown("Utforska vilka AI-bibliotek som är mest populära på GitHub")

# GitHub API-funktioner
def get_top_ai_repositories(per_page=100):
    """Hämta topp AI-repositories"""
    url = f"https://api.github.com/search/repositories?q=artificial+intelligence+OR+machine+learning+OR+deep+learning&sort=stars&order=desc&per_page={per_page}"
    response = requests.get(url)
    return response.json()

def get_repository_contents(repo_owner, repo_name, path=""):
    """Hämta innehåll från ett repository"""
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{path}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

def get_requirements_content(repo_owner, repo_name):
    """Hämta innehållet i requirements.txt eller package.json"""
    files_to_check = ["requirements.txt", "package.json", "setup.py", "environment.yml"]
    for file in files_to_check:
        contents = get_repository_contents(repo_owner, repo_name, file)
        if contents and not isinstance(contents, list):
            if "content" in contents:
                import base64
                content = base64.b64decode(contents["content"]).decode("utf-8")
                return {"file": file, "content": content}
    return None

def extract_python_libraries(content):
    """Extrahera bibliotek från requirements.txt eller setup.py"""
    libraries = []
    
    # Pattern för requirements.txt, enkla rader med namn och eventuell version
    req_pattern = r'^([a-zA-Z0-9_.-]+)(?:[=<>!~]+.*)?$'
    
    # Pattern för setup.py install_requires-listor
    setup_pattern = r'install_requires\s*=\s*\[(.*?)\]'
    
    if content.get("file") == "requirements.txt":
        for line in content.get("content", "").split("\n"):
            line = line.strip()
            if line and not line.startswith('#'):
                match = re.match(req_pattern, line)
                if match:
                    libraries.append(match.group(1).lower())
                    
    elif content.get("file") == "setup.py":
        setup_match = re.search(setup_pattern, content.get("content", ""), re.DOTALL)
        if setup_match:
            requires = setup_match.group(1)
            # Hitta alla citatdefinerade bibliotek
            lib_matches = re.finditer(r'[\'"]([a-zA-Z0-9_.-]+)[\'"]', requires)
            for match in lib_matches:
                libraries.append(match.group(1).lower())
                
    elif content.get("file") == "environment.yml":
        # Enkel parsing av environment.yml
        in_dependencies = False
        for line in content.get("content", "").split("\n"):
            line = line.strip()
            if line == "dependencies:":
                in_dependencies = True
            elif in_dependencies and line.startswith("- "):
                lib = line[2:].split("=")[0].split("<")[0].split(">")[0].strip()
                if lib and not lib.startswith("python"):
                    libraries.append(lib.lower())
                    
    elif content.get("file") == "package.json":
        try:
            import json
            package_data = json.loads(content.get("content", "{}"))
            # Kombinera både dependencies och devDependencies
            deps = package_data.get("dependencies", {})
            dev_deps = package_data.get("devDependencies", {})
            libraries = list(deps.keys()) + list(dev_deps.keys())
        except:
            pass
            
    return libraries

def is_ai_library(library):
    """Kontrollera om biblioteket är AI-relaterat"""
    ai_keywords = [
        "tensorflow", "keras", "torch", "pytorch", "scikit-learn", "sklearn", 
        "huggingface", "transformers", "spacy", "nltk", "gensim", "fastai", 
        "xgboost", "lightgbm", "catboost", "autogluon", "mxnet", "caffe", 
        "theano", "paddlepaddle", "pycaret", "ml-", "ai-", "deep-learning",
        "machine-learning", "dnn", "cnn", "rnn", "gan", "neuralnet"
    ]
    
    library = library.lower()
    return any(keyword in library for keyword in ai_keywords)

def extract_js_ai_libraries(packages):
    """Identifiera JavaScript AI-bibliotek"""
    ai_js_keywords = [
        "tensorflow", "tfjs", "ml5", "brain.js", "mind", "synaptic", 
        "machinelearn", "deeplearn", "neural", "ai-", "ml-", "neataptic",
        "classifier", "prediction", "recognition"
    ]
    
    return [pkg for pkg in packages if any(keyword in pkg.lower() for keyword in ai_js_keywords)]

def analyze_ai_libraries():
    """Analysera AI-bibliotek från topp repositories"""
    repos_data = get_top_ai_repositories(per_page=50)
    all_libraries = []
    repos_analyzed = 0
    ai_lib_count = 0
    languages = Counter()
    library_languages = {}
    
    with st.spinner("Analyserar AI-repositories... Detta kan ta lite tid."):
        if 'items' in repos_data:
            for repo in repos_data['items']:
                repos_analyzed += 1
                
                # Spara repon språk
                lang = repo.get('language', 'Unknown')
                languages[lang] += 1
                
                # Försök hämta biblioteksinformation
                owner = repo['owner']['login']
                name = repo['name']
                
                try:
                    content = get_requirements_content(owner, name)
                    if content:
                        if content['file'] in ['requirements.txt', 'setup.py', 'environment.yml']:
                            libraries = extract_python_libraries(content)
                            for lib in libraries:
                                if is_ai_library(lib):
                                    all_libraries.append(lib)
                                    ai_lib_count += 1
                                    if lib not in library_languages:
                                        library_languages[lib] = Counter()
                                    library_languages[lib][lang] += 1
                        elif content['file'] == 'package.json':
                            try:
                                import json
                                package_data = json.loads(content['content'])
                                deps = list(package_data.get('dependencies', {}).keys())
                                ai_libs = extract_js_ai_libraries(deps)
                                for lib in ai_libs:
                                    all_libraries.append(lib)
                                    ai_lib_count += 1
                                    if lib not in library_languages:
                                        library_languages[lib] = Counter()
                                    library_languages[lib][lang] += 1
                            except:
                                pass
                except Exception as e:
                    st.error(f"Fel vid analys av {owner}/{name}: {e}")
                
                # Pausa för att respektera GitHub API-begräsningar
                time.sleep(0.7)
    
    # Analysera resultaten
    library_counts = Counter(all_libraries)
    top_libraries = library_counts.most_common(20)
    
    # Skapa DataFrames för visualisering
    libs_df = pd.DataFrame(top_libraries, columns=['library', 'count'])
    
    # Skapa DataFrame för språk för varje bibliotek
    lib_lang_data = []
    for lib, count in top_libraries[:10]:  # De 10 populäraste biblioteken
        for lang, lang_count in library_languages.get(lib, {}).items():
            lib_lang_data.append({
                'library': lib,
                'language': lang,
                'count': lang_count
            })
    
    lib_lang_df = pd.DataFrame(lib_lang_data)
    langs_df = pd.DataFrame(list(languages.items()), columns=['language', 'count'])
    
    return {
        'libraries': libs_df,
        'languages': langs_df,
        'library_languages': lib_lang_df,
        'repos_analyzed': repos_analyzed,
        'ai_libraries_found': ai_lib_count
    }

# Caching av API-anrop för att respektera GitHub-begränsningar
@st.cache_data(ttl=3600)
def load_github_ai_data():
    return analyze_ai_libraries()

# Ladda data
try:
    data = load_github_ai_data()
    
    # Visa statistik
    st.subheader("AI-bibliotek Översikt")
    
    # Skapa statistikrutor
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Repositories Analyserade", data['repos_analyzed'])
    with col2:
        st.metric("AI-bibliotek hittade", data['ai_libraries_found'])
    with col3:
        if len(data['libraries']) > 0:
            st.metric("Populäraste biblioteket", data['libraries'].iloc[0]['library'])
    
    # Skapa två kolumner för visualiseringar
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Populäraste AI-biblioteken")
        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(data['libraries'].head(15)['library'], data['libraries'].head(15)['count'], color='skyblue')
        plt.xlabel('Antal förekomster')
        plt.ylabel('Bibliotek')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("Programmeringsspråk i AI-repositories")
        fig, ax = plt.subplots(figsize=(10, 8))
        langs_df = data['languages'].sort_values('count', ascending=False).head(10)
        bars = ax.barh(langs_df['language'], langs_df['count'], color='lightgreen')
        plt.xlabel('Antal repositories')
        plt.ylabel('Språk')
        plt.tight_layout()
        st.pyplot(fig)
    
    # Visa detaljerad biblioteksdata
    st.subheader("Detaljerad AI-biblioteksstatistik")
    st.dataframe(data['libraries'])
    
    # Språkanvändning per bibliotek
    st.subheader("Programmeringsspråk per AI-bibliotek")
    library_to_show = st.selectbox("Välj bibliotek", data['libraries']['library'].head(10).tolist())
    
    if library_to_show:
        lib_langs = data['library_languages'][data['library_languages']['library'] == library_to_show]
        if not lib_langs.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            lib_langs = lib_langs.sort_values('count', ascending=False)
            bars = ax.barh(lib_langs['language'], lib_langs['count'], color='coral')
            plt.xlabel('Antal förekomster')
            plt.ylabel('Språk')
            plt.title(f'Användning av {library_to_show} i olika programmeringsspråk')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info(f"Ingen språkdata tillgänglig för {library_to_show}")
    
except Exception as e:
    st.error(f"Ett fel uppstod: {e}")
    st.info("GitHub API har begränsningar för antal anrop. Vänta en stund eller använd API-nycklar för högre kvot.")

# Footer
st.markdown("---")
st.markdown("Data hämtad från GitHub API. Uppdateras varje timme. Notera att analysen baseras på innehållet i requirements.txt, package.json, setup.py, och environment.yml-filer i de 50 mest populära AI-repositoriesen.")