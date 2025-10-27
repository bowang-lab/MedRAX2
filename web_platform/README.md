# MedRAX Web Platform

A comprehensive web platform for medical imaging analysis using AI-powered tools.

## Quick Start

### Prerequisites

**Option A: Conda (Recommended for Research Servers)**

```bash
conda --version
# If not installed: https://docs.conda.io/en/latest/miniconda.html
```

**Option B: Python 3.11 (Local Development)**

```bash
python3.11 --version
# Should show: Python 3.11.x
```

If you don't have Python 3.11:
- **macOS**: `brew install python@3.11`
- **Ubuntu**: `sudo apt install python3.11`
- **Windows**: Download from python.org

### Installation

**For Conda Users (Research Servers):**

```bash
# One-time setup
cd web_platform/backend
conda env create -f environment.yml
conda activate alankrit-medrax2

# Run backend (from repo root)
cd ../..
./web_platform/start-backend.sh
```

**For Venv Users (Local Dev):**

1. **Start Backend:**
   ```bash
   cd web_platform
   ./start-backend.sh
   ```
   First run: 10-20 minutes (installs ~5-10GB)

2. **Start Frontend** (in another terminal):
   ```bash
   ./start-frontend.sh
   ```

3. **Open Browser:**
   ```
   http://localhost:3000
   ```

> **Note:** `start-backend.sh` auto-detects conda and uses it if available, otherwise falls back to Python venv.

## Python Version Requirements

| Version | Status |
|---------|--------|
| **3.11.x** | **RECOMMENDED** |
| 3.12.x | Works (some packages limited) |
| 3.13.x | NOT SUPPORTED (packages missing) |

**Why Python 3.11?**
- All 15 medical imaging tools fully supported
- All dependencies available
- Stable and well-tested
- Best performance for our use case

**Note:** Using Python 3.12+ or 3.13 will cause package installation failures.

## Features

### 15 Medical Imaging Tools

1. **Classification** (2 tools)
   - TorchXRayVision Classifier
   - ArcPlus Classifier

2. **Visual Question Answering** (3 tools)
   - CheXagent VQA
   - LLaVA-Med
   - MedGemma VQA

3. **Segmentation** (2 tools)
   - MedSAM2
   - Chest X-Ray Segmentation

4. **Generation** (2 tools)
   - Radiology Report Generator
   - X-Ray Generator

5. **Grounding** (1 tool)
   - X-Ray Phrase Grounding

6. **Processing** (1 tool)
   - DICOM Processor

7. **Retrieval** (3 tools)
   - Medical Knowledge RAG
   - DuckDuckGo Search
   - Web Browser

8. **Execution** (1 tool)
   - Python Sandbox

### Tool Management UI

- Load/Unload tools dynamically
- View tool status and dependencies
- Category-grouped organization
- Real-time status updates
- Installation guidance

### Model Caching

- Download models once, use forever
- Three cache locations:
  - HuggingFace: `~/.cache/huggingface/`
  - Torch: `~/.cache/torch/`
  - Custom: `./model_cache/`

## System Requirements

### Minimum:
- Python 3.11
- 16GB RAM
- 20GB disk space

### Recommended:
- Python 3.11.8
- 32GB RAM
- CUDA GPU with 16GB VRAM
- 50GB disk space

## Documentation

- [Conda Setup Guide](backend/CONDA_SETUP.md) - **Conda environment setup for research servers**
- [Tool Analysis](docs/MEDRAX_TOOLS_ANALYSIS.md) - Detailed tool information
- [Backend Implementation](docs/OPTION_C_IMPLEMENTATION.md) - Architecture details
- [Tool Manager](docs/TOOL_MANAGER_IMPLEMENTATION.md) - Tool loading system
- [Test Suite](backend/tests/README.md) - Testing documentation
- [Frontend Docs](frontend/docs/README.md) - Frontend architecture

## Troubleshooting

### Conda: Environment Already Exists

```bash
# Option 1: Use existing environment
conda activate alankrit-medrax2

# Option 2: Remove and recreate
conda env remove -n alankrit-medrax2
cd web_platform/backend
conda env create -f environment.yml
```

### Conda: Package Not Found

Some packages aren't available on all conda channels. They're automatically installed via pip from `requirements.txt`.

### Python 3.13 Issues (Venv Users)

If you're seeing package installation errors, you're likely using Python 3.13.

**Solution:**
```bash
# Install Python 3.11
brew install python@3.11  # macOS

# Remove old venv
cd web_platform/backend
rm -rf venv

# Create new venv with Python 3.11
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Common Issues

**Issue:** SimpleITK not found  
**Solution:** Use Python 3.11 (3.13 not supported)

**Issue:** Torch version conflicts  
**Solution:** Fresh venv with Python 3.11

**Issue:** Numpy version errors  
**Solution:** Use Python 3.11, requirements.txt updated

## Testing

```bash
cd backend
source venv/bin/activate
python -m pytest tests/ -v
```

Expected: 169/169 tests passing (100%)

## Architecture

### Backend:
- FastAPI for REST API
- SQLAlchemy for database
- JWT authentication
- Server-Sent Events for streaming

### Frontend:
- Next.js 14
- React with TypeScript
- Tailwind CSS
- Zustand for state management

### Database:
- SQLite for development
- PostgreSQL ready for production

## Development

### Backend:
```bash
cd web_platform/backend
source venv/bin/activate
uvicorn app.main:app --reload
```

### Frontend:
```bash
cd web_platform/frontend
npm run dev
```

## Production Deployment

1. Update `.env` files with production settings
2. Use Python 3.11
3. Install all dependencies
4. Run migrations
5. Start services with production config

## License

See LICENSE file for details.

## Support

For issues and questions:
- Check documentation files
- Review troubleshooting section
- Verify Python 3.11 is being used

---

**Status:** Production Ready  
**Python Version:** 3.11.x (REQUIRED)  
**Tests:** 169/169 passing (100%)  
**Last Updated:** October 20, 2025
