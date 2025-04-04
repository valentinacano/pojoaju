import os
import cv2
import numpy as np
import pandas as pd
from ml.utils.general import mediapipe_detection


# CREATE KEYPOINTS
def extract_keypoints(results):
    pose = (
        np.array(
            [
                [res.x, res.y, res.z, res.visibility]
                for res in results.pose_landmarks.landmark
            ]
        ).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 4)
    )
    face = (
        np.array(
            [[res.x, res.y, res.z] for res in results.face_landmarks.landmark]
        ).flatten()
        if results.face_landmarks
        else np.zeros(468 * 3)
    )
    lh = (
        np.array(
            [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
        ).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )
    rh = (
        np.array(
            [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
        ).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )
    return np.concatenate([pose, face, lh, rh])


def get_keypoints(model, sample_path):
    """
    ### OBTENER KEYPOINTS DE LA MUESTRA
    Retorna la secuencia de keypoints de la muestra
    """
    kp_seq = np.array([])
    for img_name in os.listdir(sample_path):
        img_path = os.path.join(sample_path, img_name)
        frame = cv2.imread(img_path)
        results = mediapipe_detection(frame, model)
        kp_frame = extract_keypoints(results)
        kp_seq = np.concatenate(
            [kp_seq, [kp_frame]] if kp_seq.size > 0 else [[kp_frame]]
        )
    return kp_seq


def insert_keypoints_sequence(df, n_sample: int, kp_seq):
    """
    ### INSERTA LOS KEYPOINTS DE LA MUESTRA AL DATAFRAME
    Retorna el mismo DataFrame pero con los keypoints de la muestra agregados
    """
    for frame, keypoints in enumerate(kp_seq):
        data = {"sample": n_sample, "frame": frame + 1, "keypoints": [keypoints]}
        df_keypoints = pd.DataFrame(data)
        df = pd.concat([df, df_keypoints])

    return df
