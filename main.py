import os

from ml.features.capture_samples import capture_samples
from app.config import ROOT_PATH, FRAME_ACTIONS_PATH

if __name__ == "__main__":
    word_name = "buenos_dias"
    word_path = os.path.join(ROOT_PATH, FRAME_ACTIONS_PATH, word_name)
    capture_samples(word_path)
