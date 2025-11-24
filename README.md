# Semantic Codebase Indexing

A production-ready semantic search system that transforms your codebase into searchable vector embeddings. This system enables AI coding assistants to understand and search your code by meaning rather than just keywords, dramatically improving code discovery and context retrieval.

> **Note**: This is a fork of [Modal](https://github.com/OJamals/Modal) by OJamals, maintained for internal use with additional features and updates.

## What This Does

This project creates a complete semantic search infrastructure for your codebase:

1. **Embedding Generation**: Uses Qwen3-0.6B (via Ollama) to convert code and text into 1024-dimensional vectors that capture semantic meaning
2. **Vector Storage**: Stores embeddings in Qdrant, a high-performance vector database optimized for similarity search
3. **OpenAI-Compatible API**: Provides a FastAPI server that mimics OpenAI's embedding API, making it compatible with any tool that supports OpenAI embeddings
4. **Restart Safety**: Fully idempotent setup that preserves data and prevents process conflicts across restarts

**Use Case**: Perfect for AI coding assistants (RooCode, Cline, Cursor, etc.) that need to search large codebases semantically to provide relevant context for code generation and analysis.

## Why This Fork?

- **Restart Safety**: Added robust process management and data preservation (original had data loss issues)
- **Production Ready**: Improved error handling, logging, and recovery mechanisms
- **Active Maintenance**: Ongoing updates and improvements
- **Model Agnostic**: Works with any embedding model via Ollama, not just Qwen3

## Quick Start

```bash
./setup.sh
```

This single command:
- Downloads and optimizes the Qwen3-0.6B embedding model
- Installs Python dependencies
- Starts Qdrant vector database in Docker
- Launches the OpenAI-compatible API server
- Configures the vector store with optimized settings

For detailed setup, see the [original README](README_ORIGINAL.md).

## Features

- **Semantic Code Search**: Find code by meaning, not keywords - understands context and intent
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI embeddings API
- **Flexible Models**: Works with any embedding model via Ollama (Qwen3, nomic-embed, etc.)
- **Optimized Storage**: Qdrant with HNSW indexing, scalar quantization, and memory optimization
- **IDE Integration**: Compatible with RooCode, Cline, Cursor, and any tool supporting OpenAI embeddings
- **Production Ready**: Restart-safe, data-preserving, with comprehensive error handling

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

The setup script now handles most API issues automatically. If you still have problems:

```bash
# Check if API is running
cat .qwen3-api.pid
ps -p $(cat .qwen3-api.pid)

# View API logs
tail -f qwen3-api.log

# Manually stop and restart
kill $(cat .qwen3-api.pid)
./setup.sh

# Check Ollama model
ollama list | grep qwen3
```

### Port Already in Use

The setup script automatically handles port conflicts, but if needed:

```bash
# Find process using port 8000
lsof -ti:8000

# Kill it
kill -9 $(lsof -ti:8000)

# Restart
./setup.sh
```

### Qdrant Issues

```bash
# Check Qdrant status
docker ps | grep qdrant

# View logs
docker logs qdrant

# Restart Qdrant
docker restart qdrant

# Full reset (preserves data in qdrant_storage/)
docker stop qdrant
docker rm qdrant
./setup.sh
```

### Data Loss or Corruption

```bash
# Force recreate collection (WARNING: deletes all vectors)
python3 qdrantsetup.py --force-recreate

# Or delete storage and start fresh
rm -rf qdrant_storage/
./setup.sh
```

### Script Interrupted (Ctrl+C)

The setup script has automatic cleanup. Simply re-run:

```bash
./setup.sh
```

All orphaned processes are automatically cleaned up, and existing data is preserved.

## Credits

- **Original Project**: [Modal](https://github.com/OJamals/Modal) by OJamals
- **This Fork**: Maintained for internal use with additional features

## License

This fork maintains attribution to the original Modal project by OJamals. The code is provided as-is for internal use and development.