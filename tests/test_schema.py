from src.ctr_cvr.synthetic_ads_generator import generate_synthetic_ads
from src.data_contracts.validators import validate_ads_schema


def test_synthetic_ads_schema_passes():
    df = generate_synthetic_ads(n_rows=100, seed=1)
    result = validate_ads_schema(df)
    assert result["status"] == "PASS"
    assert result["row_count"] == 100
