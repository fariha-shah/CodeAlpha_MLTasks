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

print("1. Exporting Credit Scoring Model (All 100 Decision Trees)...")
cr = joblib.load(os.path.join(base_dir, 'CodeAlpha_CreditScoring', 'credit_scoring_model.pkl'))
cr_scaler = joblib.load(os.path.join(base_dir, 'CodeAlpha_CreditScoring', 'scaler.pkl'))

trees = []
for dt in cr.estimators_:
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
print(f"Credit model exported with {len(trees)} trees. Size: {os.path.getsize(os.path.join(data_dir, 'credit_model.json'))/1024:.1f} KB")

print("\n2. Exporting Handwritten Digit MLP Model & Safe Scaler...")
hw = joblib.load(os.path.join(base_dir, 'CodeAlpha_HandwrittenRecognition', 'handwriting_model.pkl'))
hw_scaler = joblib.load(os.path.join(base_dir, 'CodeAlpha_HandwrittenRecognition', 'scaler.pkl'))

# Clamp scale so outer zero-variance pixels don't blow up forward pass
safe_scale = np.maximum(hw_scaler.scale_.astype(np.float32), 0.05)

hw_floats = np.concatenate([
    hw_scaler.mean_.astype(np.float32),
    safe_scale,
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

print("\n3. Exporting Emotion Recognition Model & Benchmarks...")
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

print("\nALL EXPORTS REFRESHED!")
