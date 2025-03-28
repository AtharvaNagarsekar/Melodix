import pickle 
import streamlit as st 
import pandas as pd 
import spotipy 
import random 
import nltk 
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer 
from sklearn.metrics.pairwise import cosine_similarity 
from spotipy.oauth2 import SpotifyClientCredentials 
import urllib.parse


st.set_page_config(
    page_title="Melodix | Music Discovery", 
    page_icon="🎧", 
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
:root {
    --bg-primary: #121212;
    --text-primary: #ffffff;
    --accent-color: #1DB954;
    --card-bg: #1E1E1E;
}
body {
    font-family: 'Poppins', sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
}
.stApp { 
    background-color: var(--bg-primary); 
}
.stSidebar {
    background-color: var(--card-bg) !important;
    color: var(--text-primary) !important;
}
.main-container { 
    max-width: 1200px; 
    margin: 0 auto; 
    padding: 2rem; 
}
.stTitle { 
    color: var(--text-primary); 
    text-align: center; 
    font-weight: 600; 
    margin-bottom: 1.5rem; 
    background: linear-gradient(90deg, var(--accent-color), #ffffff); 
    -webkit-background-clip: text; 
    -webkit-text-fill-color: transparent; 
}
.stSelectbox > div > div { 
    background-color: var(--card-bg) !important; 
    color: var(--text-primary) !important; 
    border: 1px solid var(--accent-color); 
}
.stButton>button { 
    background-color: var(--accent-color); 
    color: var(--bg-primary); 
    border: none; 
    border-radius: 8px; 
    padding: 10px 20px; 
    transition: all 0.3s ease; 
}
.stButton>button:hover { 
    background-color: white; 
    color: var(--bg-primary); 
}
.mood-buttons { 
    display: flex; 
    justify-content: center; 
    gap: 1rem; 
    margin-bottom: 1rem; 
}
.mood-button { 
    background-color: var(--card-bg); 
    border: 2px solid var(--accent-color); 
    color: var(--text-primary); 
    padding: 10px 20px; 
    border-radius: 8px; 
    cursor: pointer; 
    transition: all 0.3s ease; 
}
.mood-button:hover, .mood-button.active { 
    background-color: var(--accent-color); 
    color: var(--bg-primary); 
}
.recommendation-card {
    background-color: var(--card-bg);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    transition: transform 0.3s ease;
    margin-bottom: 1rem;
}
.recommendation-card img {
    border-radius: 8px;
    width: 100%;
    aspect-ratio: 1/1;
    object-fit: cover;
}
.recommendation-card h4 {
    color: white;
    margin: 10px 0 5px 0;
}
.recommendation-card p {
    color: #1DB954;
    margin-bottom: 10px;
}
.spotify-button {
    background-color: #1DB954;
    color: white;
    border: none;
    border-radius: 20px;
    padding: 8px 15px;
    text-decoration: none;
    display: inline-block;
    margin-top: 5px;
}
.footer {
    margin-top: 3rem;
    padding: 2rem;
    background-color: var(--card-bg);
    border-radius: 12px;
    text-align: center;
}
.footer-content {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
    margin-bottom: 1rem;
}
.history-item {
    padding: 0.5rem;
    border-bottom: 1px solid #333;
}
.history-item:last-child {
    border-bottom: none;
}
</style>
""", unsafe_allow_html=True)

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)
CLIENT_ID = st.secrets["CLIENT_ID"]
CLIENT_SECRET = st.secrets["CLIENT_SECRET"]

client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
def tokenization(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(str(text).lower())
    return " ".join([lemmatizer.lemmatize(token) for token in tokens])

def get_song_album_cover_url(song_name, artist_name):
    try:
        search_query = f"track:{song_name} artist:{artist_name}"
        results = sp.search(q=search_query, type="track", limit=1)
        if results and results["tracks"]["items"]:
            return results["tracks"]["items"][0]["album"]["images"][0]["url"]
    except:
        pass
    return "https://i.postimg.cc/0QNxYz4V/social.png"
def get_spotify_link(song_name, artist_name):
    try:
        search_query = f"track:{song_name} artist:{artist_name}"
        results = sp.search(q=search_query, type="track", limit=1)
        
        if results['tracks']['items']:
            track = results['tracks']['items'][0]
            return track['external_urls']['spotify']
    except:
        pass
    
    search_query = urllib.parse.quote(f"{song_name} {artist_name}")
    return f"https://open.spotify.com/search/{search_query}"
@st.cache_resource
def load_model():
    with open("song_recommendation_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
    df = model["data"]
    all_songs = sorted(df["song"].unique().tolist())
except:
    st.error("Failed to load recommendation model. Please ensure the model file exists.")
    st.stop()

MOOD_KEYWORDS = {
    "🌞 Happy": ["happy", "joy", "upbeat", "energetic", "positive"],
    "😴 Chill": ["relax", "calm", "peaceful", "smooth"],
    "💔 Sad": ["sad", "melancholy", "heartbreak", "emotional"],
    "🔥 Energetic": ["party", "dance", "workout", "exciting"]
}

def recommend(query, mood=None, top_n=5):
    try:
        tfidf_v = model["tfidf"]
        
        if mood:
            mood_keywords = " ".join(MOOD_KEYWORDS.get(mood, []))
            query = f"{query} {mood_keywords}"
        
        query_processed = tokenization(query)
        query_vec = tfidf_v.transform([query_processed]).toarray()
        query_sim = cosine_similarity(query_vec, tfidf_v.transform(df["text"]).toarray()).flatten()
        
        top_indices = query_sim.argsort()[::-1]
        recommended_songs = df.iloc[top_indices][["song", "artist"]]
        recommended_songs = recommended_songs[recommended_songs["song"].str.lower() != query.lower()]
        recommended_songs = recommended_songs.drop_duplicates(subset=["song"]).head(top_n)
        
        recommendations_list = []
        for _, row in recommended_songs.iterrows():
            recommendations_list.append({
                'Song': row['song'],
                'Artist': row['artist'],
                'Album Cover': get_song_album_cover_url(row['song'], row['artist']),
                'Spotify Link': get_spotify_link(row['song'], row['artist'])
            })
        
        return pd.DataFrame(recommendations_list)
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        return pd.DataFrame()
def main():
    if 'selected_song' not in st.session_state:
        st.session_state.selected_song = random.choice(all_songs)
    
    if 'recommendation_history' not in st.session_state:
        st.session_state.recommendation_history = []

    with st.sidebar:
        st.markdown("## 🎵 Melodix")
        st.markdown("**Music Recommendation System**")
        
        st.markdown("### 📜 Your History")
        if st.session_state.recommendation_history:
            for item in st.session_state.recommendation_history[:5]: 
                st.markdown(f"""
                <div class="history-item">
                    <p><strong>{item['song']}</strong></p>
                    <p>Mood: {item['mood']}</p>
                    <p><small>{item['time']}</small></p>
                </div>
                """, unsafe_allow_html=True)
            
            if st.button("Clear History"):
                st.session_state.recommendation_history = []
                st.success("History cleared!")
        else:
            st.markdown("<p>Your recommendation history will appear here</p>", unsafe_allow_html=True)
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="stTitle">Melodix | Music Discovery</h1>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    
    with col1:
        mood = st.radio(
            "Select Your Mood", 
            list(MOOD_KEYWORDS.keys()), 
            horizontal=True
        )
    
    with col2:
        st.session_state.selected_song = st.selectbox(
            "Choose a song", 
            options=all_songs, 
            index=all_songs.index(st.session_state.selected_song)
        )
    
    if st.button("Discover Music", type="primary"):
        with st.spinner('Finding similar tracks...'):
            recommendations = recommend(st.session_state.selected_song, mood)
            
            if not recommendations.empty:
                st.session_state.recommendation_history.insert(0, {
                    'song': st.session_state.selected_song,
                    'mood': mood,
                    'time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')
                })
                st.markdown("### 🎧 Recommendations")
                cols = st.columns(len(recommendations))
                
                for idx, (_, row) in enumerate(recommendations.iterrows()):
                    with cols[idx]:
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <a href="{row['Spotify Link']}" target="_blank">
                                <img src="{row['Album Cover']}" alt="{row['Song']}">
                            </a>
                            <h4>{row['Song']}</h4>
                            <p>{row['Artist']}</p>
                            <a href="{row['Spotify Link']}" target="_blank" class="spotify-button">
                                Play on Spotify
                            </a>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("No recommendations found. Try a different song.")
    st.markdown("""
    <style>
        .footer {
            text-align: center;
            padding: 20px;
        }
        .footer-content {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .footer h3, .footer p {
            text-align: center;
        }
        hr {
            width: 80%;
            border-color: #333;
            margin: 1.5rem auto;
        }
    </style>

    <div class="footer">
        <div class="footer-content">
            <h3>How It Works</h3>
            <p>Melodix dances through Spotify's vast musical universe, using NLTK's linguistic alchemy to decode the hidden poetry in song lyrics. Our Scikit-learn powered recommendation engine then maps these musical fingerprints, finding harmonious connections between tracks you love and undiscovered gems. By analyzing the DNA of songs through advanced TF-IDF algorithms, we curate playlists that feel personally tuned to your taste. The result? Uncannily accurate recommendations that evolve with your musical journey.</p>
        </div>
        <hr>
        <p style="color: var(--accent-color); font-style: italic;">
            Made with ❤️ by a music lover - Atharva
        </p>
    </div>
""", unsafe_allow_html=True)




if __name__ == "__main__":
    main()