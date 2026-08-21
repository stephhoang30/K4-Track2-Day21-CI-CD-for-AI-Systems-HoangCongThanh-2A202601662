from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import joblib
import os

app = FastAPI()

ARTIFACT_BUCKET = os.environ["ARTIFACT_BUCKET"]
MODEL_KEY = os.environ.get("MODEL_KEY", "artifacts/current/model.joblib")
MODEL_PATH = os.path.expanduser("~/models/model.joblib")

LABELS = {0: "thu_nhap_thap", 1: "thu_nhap_cao"}


def download_model():
    """
    Tai file model.joblib tu cloud storage ve may khi server khoi dong.

    Ham nay duoc goi mot lan khi module duoc import. boto3 tu tim credentials
    theo thu tu: bien moi truong -> ~/.aws/credentials -> IAM role cua EC2
    instance. Cach sach nhat tren EC2 la gan IAM instance profile, khong can
    copy key nao len VM.
    """
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    s3 = boto3.client("s3")
    s3.download_file(ARTIFACT_BUCKET, MODEL_KEY, MODEL_PATH)

    size_mb = os.path.getsize(MODEL_PATH) / 1024 / 1024
    print(f"Da tai model tu s3://{ARTIFACT_BUCKET}/{MODEL_KEY} ({size_mb:.2f} MB)")


download_model()
model = joblib.load(MODEL_PATH)


class ScoreRequest(BaseModel):
    features: list[float]


@app.get("/healthz")
def healthz():
    """
    Endpoint kiem tra suc khoe server.
    GitHub Actions goi endpoint nay sau khi deploy de xac nhan server dang chay.
    """
    return {"status": "ok"}


@app.post("/score")
def score(req: ScoreRequest):
    """
    Endpoint suy luan chinh.

    Dau vao : JSON {"features": [f1, f2, ..., f10]}
    Dau ra  : JSON {"prediction": <0|1>, "label": <"thu_nhap_thap"|"thu_nhap_cao">}

    Thu tu 10 dac trung:
        age, workclass, education_num, marital_status, occupation,
        relationship, sex, capital_gain, capital_loss, hours_per_week
    """
    if len(req.features) != 10:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 10 features (adult income), got {len(req.features)}",
        )

    pred = int(model.predict([req.features])[0])
    return {"prediction": pred, "label": LABELS[pred]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
