from ml.features.pipelines import create_samples_from_camera
from app.config import FRAME_ACTIONS_PATH, KEYPOINTS_PATH


if __name__ == "__main__":
    word_name = "buenos_dias"

    create_samples_from_camera(
        word_name=word_name, root_path=FRAME_ACTIONS_PATH, keypoints_path=KEYPOINTS_PATH
    )
