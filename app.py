import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import time
from audio_recorder_streamlit import audio_recorder
from emotion_model import EmotionRecognizer

# Initialize emotion recognizer
def load_model():
    try:
        recognizer = EmotionRecognizer()
        st.session_state['model_loaded'] = True
        return recognizer
    except Exception as e:
        st.error(f"❌ Failed to load emotion model: {str(e)}")
        st.session_state['model_loaded'] = False
        return None

if 'recognizer' not in st.session_state:
    st.session_state['recognizer'] = load_model()

# App configuration
st.set_page_config(
    page_title="Voice Emotion Detection",
    page_icon="🎙️",
    layout="centered"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-container {
        max-width: 800px;
        padding: 2rem;
    }
    .recording-btn {
        background-color: #4CAF50 !important;
        color: white !important;
        border: none !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
        font-size: 16px !important;
        transition: all 0.3s !important;
    }
    .recording-btn:hover {
        background-color: #45a049 !important;
        transform: scale(1.02);
    }
    .result-card {
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .progress-container {
        height: 10px;
        border-radius: 5px;
        background-color: #e0e0e0;
        margin: 1rem 0;
    }
    .progress-bar {
        height: 100%;
        border-radius: 5px;
        background: linear-gradient(90deg, #4CAF50 0%, #FFC107 50%, #F44336 100%);
        transition: width 0.5s;
    }
    .emotion-emoji {
        font-size: 3rem;
        margin: 0.5rem 0;
    }
    .error-box {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .feature-plot {
        margin-top: 1rem;
    }
    .retry-btn {
        background-color: #2196F3 !important;
        margin-top: 1rem;
    }
    .sample-audio-btn {
        background-color: #9C27B0 !important;
        margin-bottom: 1rem;
    }
    .debug-info {
        font-family: monospace;
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def save_audio_file(audio_bytes):
    """Save audio with proper format and validation"""
    os.makedirs("temp_audio", exist_ok=True)
    temp_path = "temp_audio/recording.wav"
    
    try:
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)
        
        # Verify file was saved correctly
        if os.path.getsize(temp_path) > 0:
            return temp_path
        return None
    except Exception as e:
        st.error(f"Error saving audio: {str(e)}")
        return None

def visualize_audio_features(audio_path):
    """Show detailed audio feature visualizations"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        
        # Create a 2x2 grid of plots
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        plt.tight_layout(pad=3.0)
        
        # Waveform
        librosa.display.waveshow(y, sr=sr, ax=ax[0, 0], color='#4CAF50')
        ax[0, 0].set_title('Audio Waveform', fontsize=12)
        
        # Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr, ax=ax[0, 1])
        ax[0, 1].set_title('Spectrogram', fontsize=12)
        fig.colorbar(img, ax=ax[0, 1], format="%+2.0f dB")
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        librosa.display.specshow(mfccs, x_axis='time', sr=sr, ax=ax[1, 0])
        ax[1, 0].set_title('MFCCs', fontsize=12)
        
        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax[1, 1])
        ax[1, 1].set_title('Chroma Features', fontsize=12)
        
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.warning(f"Couldn't visualize audio features: {str(e)}")

def display_emotion_results(prediction, probabilities, audio_path=None):
    """Display emotion prediction results with visualizations"""
    if prediction is None:
        st.markdown("""
        <div class='error-box'>
            <h4>❌ Analysis Failed</h4>
            <p>Possible reasons:</p>
            <ul>
                <li>Audio recording too short (need at least 3 seconds)</li>
                <li>Background noise interfering with analysis</li>
                <li>Technical issue with the model</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Try Again", key="retry_button", help="Click to record again"):
            st.experimental_rerun()
        return
    
    # Emoji mapping with color coding
    emoji_map = {
        'neutral': {'emoji': '😐', 'color': '#9E9E9E'},
        'calm': {'emoji': '😌', 'color': '#64B5F6'},
        'happy': {'emoji': '😊', 'color': '#FFD600'},
        'sad': {'emoji': '😢', 'color': '#2196F3'},
        'angry': {'emoji': '😠', 'color': '#F44336'},
        'fearful': {'emoji': '😨', 'color': '#9C27B0'},
        'disgust': {'emoji': '🤢', 'color': '#4CAF50'},
        'surprised': {'emoji': '😲', 'color': '#FF9800'}
    }
    
    emotion_data = emoji_map.get(prediction, {'emoji': '❓', 'color': '#607D8B'})
    max_prob = max(probabilities)
    
    # Main result card
    st.markdown(f"""
    <div class='result-card'>
        <div style="text-align: center;">
            <div class='emotion-emoji'>{emotion_data['emoji']}</div>
            <h2 style="color: {emotion_data['color']};">{prediction.capitalize()}</h2>
            <div class='progress-container'>
                <div class='progress-bar' style='width: {max_prob*100}%;'></div>
            </div>
            <p>Confidence: <strong>{max_prob:.1%}</strong></p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Emotion probabilities radar chart
    emotions = list(st.session_state['recognizer'].label_encoder.classes_)  # Use actual class names
    angles = np.linspace(0, 2*np.pi, len(emotions), endpoint=False).tolist()
    
    # Ensure probabilities match the number of emotions
    if len(probabilities) != len(emotions):
        st.warning(f"Probability array length ({len(probabilities)}) doesn't match number of emotions ({len(emotions)})")
        probabilities = probabilities[:len(emotions)]  # Truncate or adjust as needed
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles + angles[:1], list(probabilities) + [probabilities[0]], 'o-', linewidth=2, color=emotion_data['color'])
    ax.fill(angles + angles[:1], list(probabilities) + [probabilities[0]], alpha=0.25, color=emotion_data['color'])
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles), emotions)
    ax.set_title("Emotion Probability Distribution", pad=20)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["20%", "40%", "60%", "80%"], color="grey", size=10)
    plt.ylim(0, 1)
    
    st.pyplot(fig)
    plt.close()
    
    # Feedback and suggestions
    feedback = {
        'neutral': "You sound neutral and composed. Try adding more emotion to your voice for better detection.",
        'calm': "Your voice sounds calm and relaxed. This is great for stress-free communication!",
        'happy': "You sound happy and cheerful! Positive emotions are contagious.",
        'sad': "You sound a bit down or sad. Remember that emotions are temporary.",
        'angry': "You sound angry or frustrated. Take a deep breath and try to relax.",
        'fearful': "You sound fearful or anxious. Focus on slow, deep breathing to calm down.",
        'disgust': "You sound disgusted or upset. Try to identify what's causing this reaction.",
        'surprised': "You sound surprised or shocked! Unexpected events can trigger strong emotions."
    }
    
    st.info(feedback.get(prediction, ""))
    
    # Debug information
    if st.session_state.get('debug', False) and audio_path:
        with st.expander("Debug Information", expanded=False):
            st.write("### Feature Extraction Pipeline")
            
            y, sr = librosa.load(audio_path, sr=16000)
            features = st.session_state['recognizer'].extract_all_features(y, sr)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("#### All Extracted Features")
                st.json({k: float(v) for k, v in features.items()})
            
            with col2:
                st.write("#### Selected Features")
                selected_features = {k: features[k] for k in st.session_state['recognizer'].required_features}
                st.dataframe(pd.DataFrame.from_dict(selected_features, orient='index', columns=['Value']))
            
            st.write("#### Feature Statistics")
            st.dataframe(pd.DataFrame({
                'Mean': np.mean(list(features.values())),
                'Std': np.std(list(features.values())),
                'Min': np.min(list(features.values())),
                'Max': np.max(list(features.values()))
            }, index=['All Features']))

def main():
    st.title("🎙️ Voice Emotion Recognition")
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p>Speak naturally for 3-5 seconds after clicking the record button.</p>
        <p>Try expressing different emotions like happiness, sadness, or anger.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Recording section
    with st.expander("🎤 Voice Recorder", expanded=True):
        audio_bytes = audio_recorder(
            text="Click to record",
            recording_color="#F44336",
            neutral_color="#4CAF50",
            icon_name="microphone",
            icon_size="2x",
            pause_threshold=3.0,
            sample_rate=16000
        )
        
        if audio_bytes:
            with st.spinner("Processing audio..."):
                # Create progress bar
                progress_bar = st.progress(0)
                
                # Save audio file
                audio_path = save_audio_file(audio_bytes)
                
                # Simulate processing steps
                for percent_complete in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(percent_complete + 1)
                
                if audio_path:
                    # Check audio duration
                    y, sr = librosa.load(audio_path, sr=None)
                    duration = librosa.get_duration(y=y, sr=sr)
                    
                    if duration < 1.0:
                        st.error("⚠️ Recording too short - please record at least 3 seconds of audio")
                        st.audio(audio_bytes, format="audio/wav")
                        if st.button("Try Again", key="short_retry"):
                            st.experimental_rerun()
                    else:
                        st.audio(audio_bytes, format="audio/wav")
                        
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.subheader("Audio Analysis")
                            visualize_audio_features(audio_path)
                        
                        with col2:
                            st.subheader("Emotion Detection")
                            try:
                                prediction, probabilities = st.session_state['recognizer'].predict_emotion(audio_path)
                                display_emotion_results(prediction, probabilities, audio_path)
                            except Exception as e:
                                st.error(f"Analysis failed: {str(e)}")
                                if st.button("Try Again", key="error_retry"):
                                    st.experimental_rerun()

    # Add debug toggle and model info
    st.sidebar.title("Settings")
    st.sidebar.checkbox("Show debug info", key='debug')
    
    if st.session_state.get('debug', False) and st.session_state['model_loaded']:
        st.sidebar.write("### Model Info")
        st.sidebar.write(f"Features expected: {len(st.session_state['recognizer'].required_features)}")
        st.sidebar.write(f"Classes: {list(st.session_state['recognizer'].emotion_map.values())}")

    # Add footer with information
    st.markdown("""
    <div style="text-align: center; margin-top: 3rem; color: #666;">
        <p>For best results, record in a quiet environment and speak clearly.</p>
        <p>The model works best with 3-5 seconds of clear speech.</p>
        <p><small>Note: This is a machine learning model and may not always be accurate.</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()