import os
import shutil
import glob
import json
import joblib
import numpy as np

base_dir = r'D:\CodeAlpha-Tasks'
data_dir = os.path.join(base_dir, 'data')
audio_dir = os.path.join(base_dir, 'assets', 'audio')
os.makedirs(data_dir, exist_ok=True)
os.makedirs(audio_dir, exist_ok=True)

print("1. Exporting Credit Scoring Model...")
cr = joblib.load(os.path.join(base_dir, 'CodeAlpha_CreditScoring', 'credit_scoring_model.pkl'))
cr_scaler = joblib.load(os.path.join(base_dir, 'CodeAlpha_CreditScoring', 'scaler.pkl'))

trees = []
# Use 60 trees for great speed & precision
for dt in cr.estimators_[:60]:
    tree = dt.tree_
    trees.append({
        'feature': tree.feature.tolist(),
        'threshold': np.round(tree.threshold, 4).tolist(),
        'children_left': tree.children_left.tolist(),
        'children_right': tree.children_right.tolist(),
        'value': np.round(tree.value[:, 0, :], 4).tolist()
    })

credit_data = {
    'features': ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Credit amount', 'Duration', 'Purpose'],
    'numeric_indices': [0, 6, 7], # Age, Credit amount, Duration
    'scaler_mean': [float(x) for x in cr_scaler.mean_],
    'scaler_scale': [float(x) for x in cr_scaler.scale_],
    'feature_importances': [float(x) for x in cr.feature_importances_],
    'classes': [int(c) for c in cr.classes_], # 0 = bad, 1 = good
    'encoders': {
        'Sex': {'female': 0, 'male': 1},
        'Housing': {'free': 0, 'own': 1, 'rent': 2},
        'Saving accounts': {'little': 0, 'moderate': 1, 'quite rich': 2, 'rich': 3, 'unknown': 4},
        'Checking account': {'little': 0, 'moderate': 1, 'rich': 2, 'unknown': 3},
        'Purpose': {
            'business': 0,
            'car': 1,
            'domestic appliances': 2,
            'education': 3,
            'furniture/equipment': 4,
            'radio/TV': 5,
            'repairs': 6,
            'vacation/others': 7
        },
        'Risk': {'bad': 0, 'good': 1}
    },
    'trees': trees,
    'metrics': {
        'accuracy': 74.50,
        'roc_auc': 0.753,
        'bad_precision': 0.69,
        'bad_recall': 0.55,
        'good_precision': 0.76,
        'good_recall': 0.86
    }
}

with open(os.path.join(data_dir, 'credit_model.json'), 'w', encoding='utf-8') as f:
    json.dump(credit_data, f)
print(f"Credit model exported. File size: {os.path.getsize(os.path.join(data_dir, 'credit_model.json'))/1024:.1f} KB")

print("\n2. Exporting Handwritten Digit MLP Model...")
hw = joblib.load(os.path.join(base_dir, 'CodeAlpha_HandwrittenRecognition', 'handwriting_model.pkl'))
hw_scaler = joblib.load(os.path.join(base_dir, 'CodeAlpha_HandwrittenRecognition', 'scaler.pkl'))

# Pack into Float32Array binary:
# [scaler_mean (784), scaler_scale (784), W0 (784*256), b0 (256), W1 (256*128), b1 (128), W2 (128*10), b2 (10)]
hw_floats = np.concatenate([
    hw_scaler.mean_.astype(np.float32),
    hw_scaler.scale_.astype(np.float32),
    hw.coefs_[0].astype(np.float32).flatten(),
    hw.intercepts_[0].astype(np.float32),
    hw.coefs_[1].astype(np.float32).flatten(),
    hw.intercepts_[1].astype(np.float32),
    hw.coefs_[2].astype(np.float32).flatten(),
    hw.intercepts_[2].astype(np.float32)
])

bin_path = os.path.join(data_dir, 'handwriting_weights.bin')
hw_floats.tofile(bin_path)
print(f"Handwriting weights exported to {bin_path} ({os.path.getsize(bin_path)/1024:.1f} KB)")

# Also write metadata
hw_meta = {
    'input_size': 784,
    'hidden1': 256,
    'hidden2': 128,
    'output_size': 10,
    'accuracy': 94.61,
    'classes': list(range(10))
}
with open(os.path.join(data_dir, 'handwriting_meta.json'), 'w', encoding='utf-8') as f:
    json.dump(hw_meta, f, indent=2)

print("\n3. Exporting Emotion Recognition Model...")
em = joblib.load(os.path.join(base_dir, 'CodeAlpha-EmotionRecognition', 'emotion_model.pkl'))
em_scaler = joblib.load(os.path.join(base_dir, 'CodeAlpha-EmotionRecognition', 'scaler.pkl'))
em_le = joblib.load(os.path.join(base_dir, 'CodeAlpha-EmotionRecognition', 'label_encoder.pkl'))

emotion_classes = em_le.classes_.tolist()

emotion_data = {
    'classes': emotion_classes,
    'scaler_mean': [round(float(x), 5) for x in em_scaler.mean_],
    'scaler_scale': [round(float(x), 5) for x in em_scaler.scale_],
    'W0': np.round(em.coefs_[0], 5).tolist(),
    'b0': np.round(em.intercepts_[0], 5).tolist(),
    'W1': np.round(em.coefs_[1], 5).tolist(),
    'b1': np.round(em.intercepts_[1], 5).tolist(),
    'W2': np.round(em.coefs_[2], 5).tolist(),
    'b2': np.round(em.intercepts_[2], 5).tolist(),
    'accuracy': 60.42
}

with open(os.path.join(data_dir, 'emotion_model.json'), 'w', encoding='utf-8') as f:
    json.dump(emotion_data, f)
print(f"Emotion model exported. Size: {os.path.getsize(os.path.join(data_dir, 'emotion_model.json'))/1024:.1f} KB")

print("\n4. Copying RAVDESS Audio Samples & Precomputing MFCC Features...")
import librosa

code_to_emotion = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

actor1_files = glob.glob(os.path.join(base_dir, 'CodeAlpha-EmotionRecognition', 'Actor_01', '*.wav'))
selected_audio = {}

for f in actor1_files:
    parts = os.path.basename(f).split('.')[0].split('-')
    if len(parts) >= 3:
        code = parts[2]
        emo = code_to_emotion.get(code)
        if emo and emo not in selected_audio:
            dest_file = f"{emo}.wav"
            dest_path = os.path.join(audio_dir, dest_file)
            shutil.copyfile(f, dest_path)
            
            # Compute exact MFCC & predictions
            audio, sr = librosa.load(dest_path, duration=3, offset=0.5)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            mfcc_mean = np.mean(mfcc.T, axis=0)
            mfcc_scaled = em_scaler.transform(mfcc_mean.reshape(1, -1))
            proba = em.predict_proba(mfcc_scaled)[0]
            pred_idx = int(np.argmax(proba))
            pred_label = emotion_classes[pred_idx]
            
            selected_audio[emo] = {
                'id': emo,
                'name': emo.capitalize(),
                'filename': f"assets/audio/{dest_file}",
                'actual': emo,
                'predicted': pred_label,
                'confidence': round(float(proba[pred_idx]) * 100, 2),
                'probabilities': {cls: round(float(p) * 100, 2) for cls, p in zip(emotion_classes, proba)},
                'mfcc': [round(float(v), 5) for v in mfcc_mean]
            }
            print(f"  Saved {dest_file:15} -> Actual: {emo:10} | Pred: {pred_label:10} ({proba[pred_idx]*100:.1f}%)")

with open(os.path.join(data_dir, 'audio_samples.json'), 'w', encoding='utf-8') as f:
    json.dump(selected_audio, f, indent=2)

print("\nALL EXPORTS FINISHED SUCCESSFULLY!")
