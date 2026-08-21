import os

import mlflow
import pytest


@pytest.fixture(autouse=True, scope="session")
def isolated_mlflow_store(tmp_path_factory):
    """
    Tro MLflow vao mot thu muc tam rieng cho ca phien test.

    Hai ly do:
    1. Test khong ghi run rac vao mlflow.db that - giu sach bang chung Buoc 1.
    2. Test khong phu thuoc vao trang thai ./mlruns co san tren may, nen ket qua
       tren may ca nhan va tren GitHub Actions runner la nhu nhau.

    Luu y: duong dan phai la thu muc CHUA ton tai thi MLflow moi khoi tao
    experiment mac dinh ben trong no.
    """
    store = tmp_path_factory.mktemp("mlflow") / "mlruns"
    uri = store.as_uri()
    os.environ["MLFLOW_TRACKING_URI"] = uri
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("unit-test")
    yield
