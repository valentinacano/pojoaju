import numpy as np
import pytest

from ml.split import stratified_three_way_indices


def test_split_reserva_cada_clase_en_los_tres_conjuntos():
    y = np.repeat(np.arange(3), 5)
    train, val, test = stratified_three_way_indices(y)

    assert set(y[train]) == {0, 1, 2}
    assert set(y[val]) == {0, 1, 2}
    assert set(y[test]) == {0, 1, 2}
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))


def test_split_requiere_tres_muestras_por_clase():
    with pytest.raises(ValueError, match="al menos 3"):
        stratified_three_way_indices(np.array([0, 0, 1, 1, 1]))
