from dataclasses import dataclass

@dataclass
class AdapterResult:
    name: str
    status: str
    rows: int
    message: str
    data: object = None

class BaseAdapter:
    name = "base"
    def load_raw(self, path, nrows=None):
        raise NotImplementedError
    def validate_raw(self, df):
        return {"status": "PASS", "rows": len(df), "problems": []}
    def to_standard_schema(self, df):
        raise NotImplementedError
