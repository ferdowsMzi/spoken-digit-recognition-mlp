import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
import config

def extract_features(file_path, method='mean'):
    y, sr = librosa.load(file_path, sr=None)

    frame_length = int(0.025 * sr)
    hop_length = frame_length // 2

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12, n_mels=24, n_fft=frame_length, hop_length=hop_length)

    delta = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.concatenate((mfcc, delta, delta2), axis=0)

    if method=='mean':
        return np.mean(features, axis=1)
    elif method=='pad':
        if features.shape[1] < config.MAX_PAD_LEN:
            pad_width = config.MAX_PAD_LEN - features.shape[1]
            features = np.pad(features, pad_width=((0,0), (0, pad_width)), mode='constant')
        else: 
            features = features[:, :config.MAX_PAD_LEN]
        return features.flatten()


def to_bipolar_one_hot(y, num_classes=10):
    one_hot = np.ones((len(y), num_classes)) * -1
    for i, label in enumerate(y):
        one_hot[i, label] = 1
    return one_hot

def prepare_data(method='mean'):
    cache_x = os.path.join(config.OUTPUT_DIR, f'cached_X_{method}.npy')
    cache_y = os.path.join(config.OUTPUT_DIR, f'cached_Y_{method}.npy')

    if os.path.exists(cache_x) and os.path.exists(cache_y):
        print(f'Loading {method} cached features from disk...')
        X = np.load(cache_x)
        y = np.load(cache_y)

    else:
        print(f'Extracting features using {method} method...')
        X, y = [], []

        for speaker_folder in sorted(os.listdir(config.DATA_DIR)) :
            folder_path = os.path.join(config.DATA_DIR, speaker_folder)
            if not os.path.isdir(folder_path): continue

            files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]

            if config.SAMPLES_PER_SPEAKER:
                files = files[:config.SAMPLES_PER_SPEAKER]

            for file_name in files:
                label = int(file_name.split('_')[0])
                file_path = os.path.join(folder_path, file_name)

                X.append(extract_features(file_path, method=method))
                y.append(label)

        X, y = np.array(X), np.array(y)
        np.save(cache_x, X)
        np.save(cache_y, y)
        print(f'{method} features extracted and cached!')

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)