import os

from ml.utils.normalize_samples import (
    read_frames_from_directory,
    clear_directory,
    save_normalized_frames,
    normalize_frames,
)


def normalize_samples(word_directory, target_frame_count=15):
    for sample_name in os.listdir(word_directory):
        sample_directory = os.path.join(word_directory, sample_name)
        if os.path.isdir(sample_directory):
            frames = read_frames_from_directory(sample_directory)
            normalized_frames = normalize_frames(frames, target_frame_count)
            clear_directory(sample_directory)
            save_normalized_frames(sample_directory, normalized_frames)
