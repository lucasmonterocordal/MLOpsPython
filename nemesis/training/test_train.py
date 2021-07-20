import numpy as np
from nemesis.training.train import train_model, get_model_metrics


def test_train_model():

    X_train = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
    y_train = np.array([10, 9, 8, 8, 6, 5])
    data = {"train": {"X": X_train, "y": y_train}}

    reg_model = train_model(data)

    preds = reg_model.predict([[1], [2]])
    np.testing.assert_almost_equal(preds, [10, 9])


def test_get_model_metrics():
    class MockModel:

        @staticmethod
        def predict(data):
            return ([8, 7])

    X_test = np.array([3, 4]).reshape(-1, 1)
    y_test = np.array([8, 8])
    data = {"test": {"X": X_test, "y": y_test}}

    metrics = get_model_metrics(MockModel(), data)

    assert 'accuracy' in metrics
    accuracy_score = metrics['accuracy']
    print(accuracy_score)
    np.testing.assert_almost_equal(accuracy_score, 0.5)
