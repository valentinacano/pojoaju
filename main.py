from app.config import FRAME_ACTIONS_PATH, KEYPOINTS_PATH
from app.views.flask_gui import app

from ml.features.pipelines import create_samples_from_camera


if __name__ == "__main__":
    word_name = "buenos_dias"

    modo_consola = False

    if modo_consola:
        create_samples_from_camera(
            word_name=word_name,
            root_path=FRAME_ACTIONS_PATH,
            keypoints_path=KEYPOINTS_PATH,
        )

    else:
        app.run(debug=True)
