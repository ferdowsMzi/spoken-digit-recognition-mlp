import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'numbers', 'data')

OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


SAMPLES_PER_SPEAKER = None

MAX_PAD_LEN= 150

LEARNING_RATE = 0.001
EPOCHS = 20