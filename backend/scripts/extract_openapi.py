"""
Script to extract OpenAPI JSON from the FastAPI app.

Run with: uv run python -m scripts.extract_openapi
"""
import json
import sys
from pathlib import Path

# Add current directory to path so we can import main
sys.path.append(".")

from main import app

def main():
    openapi_data = app.openapi()
    
    # Save as JSON
    output_path_json = Path("openapi.json")
    with open(output_path_json, "w") as f:
        json.dump(openapi_data, f, indent=2)
    print(f"✅ OpenAPI JSON saved to {output_path_json.absolute()}")

if __name__ == "__main__":
    main()
