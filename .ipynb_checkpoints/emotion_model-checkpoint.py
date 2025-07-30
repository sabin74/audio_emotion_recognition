import os
import librosa
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis
from librosa.feature import spectral_contrast, tonnetz

class EmotionRecognizer:
    def __init__(self, model_dir="./model"):
        try:
            self.scaler = joblib.load(os.path.join(model_dir, 'Scaler.pkl'))
            self.selector = joblib.load(os.path.join(model_dir, 'SelectKBest-40-features.pkl'))
            self.model = joblib.load(os.path.join(model_dir, 'xgboost_model.pkl'))
            self.label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
            
            self.required_features = self.selector.get_feature_names_out()
            self.emotion_classes = self.label_encoder.classes_
            print(f"Model expects {len(self.required_features)} features for {len(self.emotion_classes)} classes: {self.emotion_classes}")
            
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {str(e)}")

        # Emotion mappings
        self.emotion_map = {
            0: 'neutral', 1: 'calm', 2: 'happy',
            2: 'sad', 4: 'angry', 5: 'fearful',
            6: 'disgust', 7: 'surprised'
        }

    def extract_all_features(self, y, sr):
        """Extract all possible features before selection"""
        features = {}
        
        # Ensure we have enough samples for the FFT window
        n_fft = min(512, max(64, len(y) // 2))
        
        # 1. Basic features
        features['duration'] = librosa.get_duration(y=y, sr=sr)
        features['loudness'] = np.mean(librosa.amplitude_to_db(np.abs(y)))
        
        # 2. Voice activity
        intervals = librosa.effects.split(y, top_db=20)
        active_frames = np.sum([end-start for start, end in intervals])
        features['voice_ratio'] = active_frames / len(y) if len(y) > 0 else 0.0
        
        # 3. Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft)[0]
        features['centroid_mean'] = np.mean(spectral_centroid)
        features['centroid_std'] = np.std(spectral_centroid)
        features['centroid_skew'] = skew(spectral_centroid)
        features['centroid_kurt'] = kurtosis(spectral_centroid)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft)[0]
        features['rolloff_mean'] = np.mean(spectral_rolloff)
        features['rolloff_std'] = np.std(spectral_rolloff)
        features['rolloff_skew'] = skew(spectral_rolloff)
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft)[0]
        features['bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['bandwidth_std'] = np.std(spectral_bandwidth)
        features['bandwidth_skew'] = skew(spectral_bandwidth)
        
        # 4. MFCCs (13 coefficients + deltas)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft)
        for i in range(13):
            features[f'mfcc_{i+1}'] = np.mean(mfcc[i])
        
        # Calculate delta MFCCs for specific coefficients
        delta_mfcc = librosa.feature.delta(mfcc)
        features['mfcc_2_delta'] = np.mean(delta_mfcc[1])  # 2nd coefficient delta
        features['mfcc_4_delta'] = np.mean(delta_mfcc[3])  # 4th coefficient delta
        features['mfcc_5_delta'] = np.mean(delta_mfcc[4])  # 5th coefficient delta
        features['mfcc_7_delta'] = np.mean(delta_mfcc[6])  # 7th coefficient delta
        
        # 5. Chroma features (12)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft)
        for i in range(12):
            features[f'chroma_{i+1}'] = np.mean(chroma[i])
        
        # 6. Zero-crossing rate
        features['zcr_mean'] = np.mean(librosa.feature.zero_crossing_rate(y)[0])
        
        # 7. Spectral contrast (6)
        contrast = spectral_contrast(y=y, sr=sr, n_fft=n_fft)
        for i in range(6):
            features[f'contrast_{i+1}'] = np.mean(contrast[i])
        
        # 8. Tonnetz features (6)
        ton = tonnetz(y=y, sr=sr)
        features['tonnetz_2'] = np.mean(ton[1])  # 2nd tonnetz
        features['tonnetz_4'] = np.mean(ton[3])  # 4th tonnetz
        
        # 9. Pitch features
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, n_fft=n_fft)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        features['pitch_mean'] = np.mean(pitch_values) if len(pitch_values) > 0 else 0.0
        
        # 10. Emotion code (placeholder, will be removed by selector)
        features['emotion_code'] = 0
        
        return features

    def extract_features(self, y, sr):
        """Public method that extracts all features then selects the required ones"""
        all_features = self.extract_all_features(y, sr)
        
        # Ensure we have all required features
        missing_features = set(self.required_features) - set(all_features.keys())
        if missing_features:
            print(f"Warning: Missing features {missing_features}, filling with 0")
            for feature in missing_features:
                all_features[feature] = 0.0
        
        # Only keep features that the model expects
        filtered_features = {k: all_features[k] for k in self.required_features}
        
        print(f"Extracted {len(filtered_features)} features matching model requirements")
        return filtered_features

     def predict_emotion(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=16000, mono=True)
            duration = librosa.get_duration(y=y, sr=sr)
            
            if duration < 1.0:
                raise ValueError("Audio too short (minimum 1 second required)")
            
            y = librosa.effects.trim(librosa.effects.preemphasis(y), top_db=25)[0]
            features = self.extract_features(y, sr)
            features_df = pd.DataFrame([features], columns=self.required_features)
            features_scaled = self.scaler.transform(features_df)
            
            prediction_encoded = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Ensure probabilities match classes
            if len(probabilities) != len(self.emotion_classes):
                probabilities = probabilities[:len(self.emotion_classes)]  # Truncate if needed
            
            emotion = self.label_encoder.inverse_transform([prediction_encoded])[0]
            return emotion, probabilities
            
        except Exception as e:
            print(f"Prediction failed: {str(e)}")
            return None, None

    def get_emotion_probabilities(self, audio_path):
        emotion, probs = self.predict_emotion(audio_path)
        if emotion is None:
            return None
            
        prob_dict = {label: 0.0 for label in self.emotion_classes}
        for i, label in enumerate(self.emotion_classes):
            if i < len(probs):
                prob_dict[label] = float(probs[i])
        
        return {
            'predicted_emotion': emotion,
            'probabilities': prob_dict,
            'status': 'success'
        }