import pytest
from pydantic import ValidationError

from src.api import SessionInput


def valid_payload() -> dict:
    return {
        "Administrative": 1,
        "Administrative_Duration": 12.5,
        "Informational": 0,
        "Informational_Duration": 0.0,
        "ProductRelated": 12,
        "ProductRelated_Duration": 800.0,
        "BounceRates": 0.02,
        "ExitRates": 0.04,
        "PageValues": 4.2,
        "SpecialDay": 0.0,
        "Month": "Nov",
        "OperatingSystems": "2",
        "Browser": "2",
        "Region": "1",
        "TrafficType": "3",
        "VisitorType": "Returning_Visitor",
        "Weekend": "False",
    }


def test_api_schema_accepts_unseen_categorical_values() -> None:
    payload = valid_payload()
    payload["Region"] = "new-region"

    assert SessionInput.model_validate(payload).Region == "new-region"


def test_api_schema_rejects_invalid_rates_and_extra_fields() -> None:
    payload = valid_payload()
    payload["BounceRates"] = 1.4

    with pytest.raises(ValidationError):
        SessionInput.model_validate(payload)

    payload = valid_payload()
    payload["Converted"] = 1
    with pytest.raises(ValidationError):
        SessionInput.model_validate(payload)
