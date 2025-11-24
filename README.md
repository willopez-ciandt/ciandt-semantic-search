# Semantic Codebase Indexing

Flexible semantic search and codebase indexing with OpenAI-compatible API.

> **Note**: This is a fork of [Modal](https://github.com/OJamals/Modal) by OJamals, maintained for internal use with additional features and updates.

## Why This Fork?

- Original repository hasn't been updated in 5+ months
- Added support for internal project requirements
- Ongoing maintenance and improvements
- Model-agnostic architecture (not limited to Qwen3)

## Quick Start

```bash
./setup.sh
```

For detailed setup, see the [original README](README_ORIGINAL.md).

## Features

- **Semantic Code Search**: Find code by meaning, not keywords
- **OpenAI-Compatible API**: Easy integration with AI coding assistants
- **Flexible Models**: Works with any embedding model via Ollama
- **Optimized Storage**: Fast retrieval with Qdrant vector database
- **IDE Integration**: Compatible with RooCode, Cline, and similar tools

## Configuration

### For AI Coding Assistants (RooCode/Cline)

```yaml
# Embeddings Provider
Provider: OpenAI-compatible
Base URL: http://localhost:8000
API Key: your-super-secret-qdrant-api-key
Model: qwen3-embedding
Embedding Dimension: 1024

# Vector Database
Qdrant URL: http://localhost:6333
Qdrant API Key: your-super-secret-qdrant-api-key
Collection Name: qwen3_embedding
```

## Restart Safety

The setup script is fully idempotent and can be safely re-run:

- **Process Tracking**: API process tracked via `.qwen3-api.pid`
- **Automatic Cleanup**: Stops existing processes before starting new ones
- **Error Recovery**: Cleanup on failure (Ctrl+C safe)
- **Logging**: API output captured in `qwen3-api.log`

### Service Management

```bash
# Stop API
kill $(cat .qwen3-api.pid)
# or
pkill -f qwen3-api.py

# View logs
tail -f qwen3-api.log

# Restart everything (safe)
./setup.sh
```

## Data Persistence

Vector data is preserved across restarts:

- **Automatic Preservation**: Existing Qdrant collections are kept
- **Smart Recreation**: Only recreates if configuration is wrong
- **No Data Loss**: Safe to re-run setup.sh

### Force Recreate (Deletes Data)

```bash
python3 qdrantsetup.py --force-recreate
```

## Services

- **Embedding API**: http://localhost:8000 (OpenAI-compatible)
- **Qdrant Vector DB**: http://localhost:6333
- **Ollama**: http://localhost:11434

## Advanced Usage

### Qdrant Setup Options

```bash
# Normal setup (preserves data)
python3 qdrantsetup.py

# Skip test data
python3 qdrantsetup.py --skip-test-data

# Force recreate (WARNING: deletes all data)
python3 qdrantsetup.py --force-recreate
```

## Usage Example

```python
import requests

response = requests.post("http://localhost:8000/v1/embeddings", json={
    "input": "Your text to embed",
    "model": "qwen3-embedding",
    "encoding_format": "float"
})

embeddings = response.json()["data"][0]["embedding"]
```

## Project Structure

```
├── optimize_gguf.py          # GGUF model optimizer
├── qwen3-api.py               # OpenAI-compatible API wrapper
├── qdrantsetup.py             # Vector store configuration
├── setup.sh                   # Automated setup script
├── requirements.txt           # Python dependencies
└── test_qwen_features.py      # Feature verification tests
```

## Troubleshooting

### API Not Starting
```bash
ollama list
pkill -f qwen3-api.py
python qwen3-api.py
```

### Qdrant Issues
```bash
docker ps | grep qdrant
docker restart qdrant
docker logs qdrant
```

## Credits

- **Original Project**: [Modal](https://github.com/OJamals/Modal) by OJamals
- **This Fork**: Maintained for internal use with additional features

## License

This fork maintains attribution to the original Modal project by OJamals. The code is provided as-is for internal use and development.