from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

from predict import predict


app = FastAPI(
    title="Musanze Harvest Intelligence Service",
    description="AI-based synthetic crop yield, risk, and cluster prediction service.",
    version="1.0.0",
)


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rainfall_mm: float = Field(..., ge=600, le=1400)
    avg_temp_c: float = Field(..., ge=12, le=26)
    soil_moisture_pct: float = Field(..., ge=20, le=80)
    fertilizer_kg: float = Field(..., ge=0, le=250)
    field_size_ha: float = Field(..., ge=0.2, le=4.0)
    season: str = Field(..., pattern="^[AB]$")
    cooperative_support: int = Field(..., ge=0, le=1)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "Musanze Harvest Intelligence Service",
        "verification_code": "AI-A2-7330",
    }


@app.get("/model-info")
def model_info():
    return {
        "service": "Musanze Harvest Intelligence Service",
        "student": "RUDASINGWA Theogene",
        "registration": "25/27330",
        "verification_code": "AI-A2-7330",
        "tasks": [
            "regression",
            "classification",
            "clustering",
        ],
        "synthetic_data": True,
    }


@app.post("/predict")
def make_prediction(input_data: PredictionInput):

    try:
        result = predict(input_data.model_dump())
        return result

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except TypeError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}",
        )