import os
import librosa
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis
from tqdm import tqdm

class EmotionRecognizer:
    def __init__(self, model_dir="./model"):
        # Emotion mappings
        self.emotion_map = {
            '01': 'neutral',
            '02': 'calm',
            '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            '06': 'fearful',
            '07': 'disgust',
            '08': 'surprised'
        }
        
        # Emotion arousal categories
        self.high_arousal = ['angry', 'happy', 'fearful', 'surprised']
        self.low_arousal = ['calm', 'sad', 'disgust', 'neutral']

        try:
            # Load model components
            self.scaler = joblib.load(os.path.join(model_dir, 'Scaler.pkl'))
            self.selector = joblib.load(os.path.join(model_dir, 'SelectKBest-40-features.pkl'))
            self.model = joblib.load(os.path.join(model_dir, 'xgboost_model.pkl'))
            self.label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
            
            # Get expected feature names
            self.required_features = self.selector.get_feature_names_out()
            print(f"Model loaded successfully. Expecting {len(self.required_features)} features.")
            
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {str(e)}")

    def extract_features(self, y, sr):
        """Enhanced feature extraction matching training process"""
        features = {}
        
        # Preprocessing matching training
        y = librosa.effects.preemphasis(y)
        y = librosa.effects.trim(y, top_db=25)[0]
        rms = np.sqrt(np.mean(y**2))
        y = y / (rms + 1e-6)
        
        # Basic features
        features['duration'] = librosa.get_duration(y=y, sr=sr)
        features['loudness'] = np.mean(librosa.amplitude_to_db(np.abs(y)))
        
        # Voice activity detection
        intervals = librosa.effects.split(y, top_db=20)
        active_frames = np.sum([end-start for start, end in intervals])
        features['voice_ratio'] = active_frames / len(y)
        
        # Spectral features with temporal statistics
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        
        for name, feature in [('centroid', spectral_centroid),
                             ('rolloff', spectral_rolloff),
                             ('bandwidth', spectral_bandwidth)]:
            features[f'{name}_mean'] = np.mean(feature)
            features[f'{name}_std'] = np.std(feature)
            features[f'{name}_skew'] = skew(feature)
            features[f'{name}_kurt'] = kurtosis(feature)
        
        # MFCCs with deltas
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)
        
        for i in range(13):
            features[f'mfcc_{i+1}'] = np.mean(mfcc[i])
            features[f'mfcc_{i+1}_delta'] = np.mean(mfcc_delta[i])
        
        # Chroma and advanced features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        
        for i in range(12):
            features[f'chroma_{i+1}'] = np.mean(chroma[i])
        for i in range(7):
            features[f'contrast_{i+1}'] = np.mean(contrast[i])
        for i in range(6):
            features[f'tonnetz_{i+1}'] = np.mean(tonnetz[i])
        
        # Pitch features
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_mean = np.mean(pitches[magnitudes > np.median(magnitudes)])
        features['pitch_mean'] = pitch_mean if not np.isnan(pitch_mean) else 0
        
        # Emotion code placeholder (will be removed by selector)
        features['emotion_code'] = 0
        
        # Ensure we have all required features
        missing_features = set(self.required_features) - set(features.keys())
        if missing_features:
            print(f"Warning: Missing features {missing_features}, filling with 0")
            for feature in missing_features:
                features[feature] = 0.0
        
        # Filter to only required features
        filtered_features = {k: features[k] for k in self.required_features}
        
        return filtered_features

    def predict_emotion(self, audio_path):
        """Main prediction method with robust error handling"""
        try:
            # Load and validate audio
            y, sr = librosa.load(audio_path, sr=16000, mono=True)
            duration = librosa.get_duration(y=y, sr=sr)
            
            if duration < 0.5:
                raise ValueError("Audio too short (minimum 0.5 seconds required)")
            
            # Extract and prepare features
            features = self.extract_features(y, sr)
            features_df = pd.DataFrame([features], columns=self.required_features)
            
            # Scale features
            features_scaled = self.scaler.transform(features_df)
            
            # Predict
            prediction_encoded = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Decode prediction
            emotion = self.label_encoder.inverse_transform([prediction_encoded])[0]
            
            return emotion, probabilities
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return None, None

    def get_emotion_probabilities(self, audio_path):
        """User-friendly emotion probability output"""
        emotion, probs = self.predict_emotion(audio_path)
        if emotion is None:
            return {
                'status': 'error',
                'message': 'Failed to process audio file'
            }
            
        # Create probability dictionary
        emotion_labels = self.label_encoder.classes_
        prob_dict = {label: float(prob) for label, prob in zip(emotion_labels, probs)}
        
        return {
            'predicted_emotion': emotion,
            'probabilities': prob_dict,
            'status': 'success'
        }

    def analyze_audio_segment(self, y, sr):
        """Analyze an already loaded audio segment"""
        try:
            features = self.extract_features(y, sr)
            features_df = pd.DataFrame([features], columns=self.required_features)
            features_scaled = self.scaler.transform(features_df)
            
            prediction_encoded = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            emotion = self.label_encoder.inverse_transform([prediction_encoded])[0]
            emotion_labels = self.label_encoder.classes_
            prob_dict = {label: float(prob) for label, prob in zip(emotion_labels, probabilities)}
            
            return {
                'predicted_emotion': emotion,
                'probabilities': prob_dict,
                'status': 'success'
            }
        except Exception as e:
            print(f"Segment analysis failed: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }