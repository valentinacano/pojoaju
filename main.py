from app.config import FRAME_ACTIONS_PATH
from app.views.flask_gui import app

from ml.features.pipelines import create_samples_from_camera, save_keypoints


if __name__ == "__main__":
    word_name = "buenos_dias"
    word_id = 200

    modo_consola = False

    if modo_consola:
        # create_samples_from_camera(
        #    word_name=word_name, root_path=FRAME_ACTIONS_PATH, debug_value=True
        # )

        save_keypoints(
            word_name=word_name,
            word_id=word_id,
            root_path=FRAME_ACTIONS_PATH,
        )

    else:
        app.run()
