import os

from ml.features.capture_samples import capture_samples
from ml.features.create_keypoints import create_keypoints
from ml.features.normalize_samples import normalize_samples
from ml.utils.general import create_folder
from app.config import ROOT_PATH, FRAME_ACTIONS_PATH, KEYPOINTS_PATH


if __name__ == "__main__":
    word_name = "buenos_dias"
    word_path = os.path.join(ROOT_PATH, FRAME_ACTIONS_PATH, word_name)
    hdf_path = os.path.join(KEYPOINTS_PATH, f"{word_name}.h5")

    # 1. CAPTURE SAMPLES

    capture_samples(word_path)

    # 2. NORMALIZE SAMPLES

    normalize_samples(word_path)

    # 3. CREATE KEYPOINTS

    create_folder(KEYPOINTS_PATH)
    create_keypoints(word_name, FRAME_ACTIONS_PATH, hdf_path)
