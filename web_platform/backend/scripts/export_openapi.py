"""
Export FastAPI OpenAPI schema to a JSON file.
"""

import argparse
import json
import os
import sys
import types


def _ensure_module(name: str, attrs: dict[str, object] | None = None) -> None:
    """Create a minimal stub module if missing."""
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    if attrs:
        for key, value in attrs.items():
            setattr(module, key, value)
    sys.modules[name] = module


def _stub_optional_deps() -> None:
    """Stub heavy optional deps to allow schema export without full env."""
    _ensure_module("numpy", {"ndarray": object})
    _ensure_module("pydicom", {"dcmread": lambda *_args, **_kwargs: None})
    _ensure_module("aiofiles")
    _ensure_module("PIL")
    _ensure_module("PIL.Image", {"fromarray": lambda *_args, **_kwargs: None})


def main() -> None:
    parser = argparse.ArgumentParser(description="Export OpenAPI schema from FastAPI app.")
    parser.add_argument(
        "--out",
        default="openapi.json",
        help="Output path for the OpenAPI JSON file.",
    )
    args = parser.parse_args()

    backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if backend_root not in sys.path:
        sys.path.insert(0, backend_root)

    # Provide minimal required settings for schema export
    os.environ.setdefault("SECRET_KEY", "dev-export-secret")
    os.environ.setdefault("API_SECRET_KEY", "dev-export-api-secret")
    os.environ.setdefault("SKIP_TOOL_DEP_CHECK", "1")

    # Stub optional heavy dependencies if missing
    _stub_optional_deps()

    from app.main import app  # noqa: WPS433

    schema = app.openapi()

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    print(f"OpenAPI schema written to {out_path}")


if __name__ == "__main__":
    main()
