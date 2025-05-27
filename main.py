from app.config import FRAME_ACTIONS_PATH, words, categories
from app.views.flask_gui import app

from ml.features.pipelines import create_samples_from_camera, save_keypoints

from app.database.schema import create_words_table, create_categories_table
from app.database.database_utils import insert_words, insert_categories

if __name__ == "__main__":
    create_categories_table()
    insert_categories(categories)

    create_words_table()
    insert_words(words)

    app.run(debug=True)
