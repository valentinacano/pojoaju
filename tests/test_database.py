"""
Tests para app/database/queries.py
"""

import numpy as np
import pytest
from app.database.queries import (
    insert_word,
    insert_sample,
    insert_keypoints,
    fetch_all_words,
    fetch_word_ids_with_keypoints,
    get_word_by_name,
    word_to_id,
    fetch_keypoints_for_words,
)


def test_insert_y_fetch_palabra(clean_db):
    insert_word("hola", "saludos")
    words = fetch_all_words()
    assert any(w[1] == "hola" for w in words)


def test_get_word_by_name(clean_db):
    insert_word("gracias", "saludos")
    row = get_word_by_name("gracias")
    assert row is not None
    assert row[1] == "gracias"


def test_palabra_inexistente(clean_db):
    row = get_word_by_name("xyz_no_existe")
    assert row is None


def test_insert_keypoints_y_fetch(clean_db):
    insert_word("hola", "saludos")
    wid = word_to_id("hola")
    sample_id = insert_sample(wid)

    sequence = [np.random.rand(1662) for _ in range(15)]
    insert_keypoints(wid, sample_id, sequence)

    rows = fetch_keypoints_for_words([wid])
    assert len(rows) == 15


def test_fetch_word_ids_con_keypoints(clean_db):
    insert_word("chau", "saludos")
    wid = word_to_id("chau")
    sample_id = insert_sample(wid)
    insert_keypoints(wid, sample_id, [np.random.rand(1662) for _ in range(5)])

    ids = fetch_word_ids_with_keypoints()
    assert wid in [bytes(i) for i in ids]
