from fastapi.testclient import TestClient

from app import app
from predict import predict


client = TestClient(app)


VALID_INPUT = {
    "rainfall_mm": 900,
    "avg_temp_c": 20,
    "soil_moisture_pct": 55,
    "fertilizer_kg": 100,
    "field_size_ha": 2.0,
    "season": "A",
    "cooperative_support": 1,
}


def test_valid_prediction():
    """Test that a valid input produces a prediction."""

    result = predict(VALID_INPUT)

    assert "estimated_yield_tons" in result
    assert "high_risk_prediction" in result
    assert "high_risk_probability" in result
    assert "cluster" in result
    assert result["verification_code"] == "AI-A2-7330"


def test_missing_field():
    """Test that missing required input is rejected."""

    invalid_input = VALID_INPUT.copy()
    del invalid_input["rainfall_mm"]

    try:
        predict(invalid_input)
        assert False, "Expected ValueError was not raised"
    except ValueError as error:
        assert "Missing required fields" in str(error)


def test_wrong_type():
    """Test that an incorrect data type is rejected."""

    invalid_input = VALID_INPUT.copy()
    invalid_input["rainfall_mm"] = "heavy"

    try:
        predict(invalid_input)
        assert False, "Expected TypeError was not raised"
    except TypeError as error:
        assert "rainfall_mm must be numeric" in str(error)


def test_range_error():
    """Test that values outside allowed ranges are rejected."""

    invalid_input = VALID_INPUT.copy()
    invalid_input["rainfall_mm"] = 2000

    try:
        predict(invalid_input)
        assert False, "Expected ValueError was not raised"
    except ValueError as error:
        assert "rainfall_mm must be between 600 and 1400" in str(error)


def test_reproducibility():
    """Test that identical input produces identical predictions."""

    result1 = predict(VALID_INPUT)
    result2 = predict(VALID_INPUT)

    assert result1 == result2


def test_api_health():
    """Test the health endpoint."""

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["verification_code"] == "AI-A2-7330"


def test_api_prediction():
    """Test the FastAPI prediction endpoint."""

    response = client.post(
        "/predict",
        json=VALID_INPUT
    )

    assert response.status_code == 200

    data = response.json()

    assert "estimated_yield_tons" in data
    assert "high_risk_prediction" in data
    assert "high_risk_probability" in data
    assert "cluster" in data
    assert data["verification_code"] == "AI-A2-7330"


def test_api_missing_field():
    """Test API validation for a missing field."""

    invalid_input = VALID_INPUT.copy()
    del invalid_input["field_size_ha"]

    response = client.post(
        "/predict",
        json=invalid_input
    )

    assert response.status_code == 422


def test_api_range_error():
    """Test API validation for an out-of-range value."""

    invalid_input = VALID_INPUT.copy()
    invalid_input["rainfall_mm"] = 2000

    response = client.post(
        "/predict",
        json=invalid_input
    )

    assert response.status_code == 422