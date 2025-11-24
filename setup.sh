#!/bin/bash

# Qwen3 Embedding Setup Script
# Automates the complete setup process for RooCode integration

set -e  # Exit on any error

echo "🚀 Setting up Qwen3 Embedding for RooCode..."
echo "============================================="
# ============================================================================
# PROCESS MANAGEMENT
# ============================================================================

PID_FILE=".qwen3-api.pid"
LOG_FILE="qwen3-api.log"
API_PORT=8000

is_process_running() {
    local pid=$1
    [ -n "$pid" ] && ps -p "$pid" > /dev/null 2>&1
}

is_port_in_use() {
    lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1
}

stop_api_process() {
    echo "🔍 Checking for existing API process..."
    
    if [ -f "$PID_FILE" ]; then
        local old_pid=$(cat "$PID_FILE")
        if is_process_running "$old_pid"; then
            echo "🛑 Stopping API (PID: $old_pid)..."
            kill "$old_pid" 2>/dev/null || true
            for i in {1..10}; do
                is_process_running "$old_pid" || break
                sleep 1
            done
            is_process_running "$old_pid" && kill -9 "$old_pid" 2>/dev/null || true
        fi
        rm -f "$PID_FILE"
    fi
    
    pkill -f "python3.*qwen3-api.py" 2>/dev/null || true
    sleep 1
    
    if is_port_in_use $API_PORT; then
        local port_pid=$(lsof -ti:$API_PORT 2>/dev/null || true)
        [ -n "$port_pid" ] && kill -9 $port_pid 2>/dev/null || true
    fi
}

cleanup_on_error() {
    local exit_code=$?
    echo ""
    echo "❌ Setup failed (exit: $exit_code)"
    [ -n "$API_PID" ] && is_process_running "$API_PID" && kill "$API_PID" 2>/dev/null || true
    rm -f "$PID_FILE"
    echo "💡 Safe to re-run: ./setup.sh"
    exit $exit_code
}

trap cleanup_on_error ERR INT TERM

# ============================================================================


# Check if required files exist
if [ ! -f "requirements.txt" ]; then
    echo "❌ requirements.txt not found!"
    exit 1
fi

# Check if ollama is available
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama not found! Please install Ollama first:"
    echo "   curl -fsSL https://ollama.ai/install.sh | sh"
    exit 1
fi

# Step 1: Download and optimize GGUF model
echo "📦 Step 1: Setting up Qwen3 embedding model..."

# Check if model already exists locally
if [ ! -f "qwen3-embedding-0.6b.gguf" ]; then
    echo "🔍 GGUF model not found locally. Checking Ollama models..."
    
    # Check if model exists in Ollama
    if ! ollama list | grep -q "qwen3.*embedding"; then
        echo "📥 Downloading Qwen3-Embedding-0.6B model (Q8_0-optimized)..."
        ollama pull hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0
        
        # Wait a moment for the model to be fully registered
        sleep 2
    fi
    
    # Run GGUF optimizer to extract and optimize the model
    echo "� Optimizing GGUF model from Ollama storage..."
    if python3 optimize_gguf.py hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0 qwen3-embedding; then
        echo "✅ GGUF model optimized successfully"
    else
        echo "❌ Failed to optimize GGUF model"
        echo "Trying alternative approach..."
        
        # Alternative: create Modelfile without local GGUF
        cat > Modelfile << EOF
FROM hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0
PARAMETER num_ctx 8192
PARAMETER embedding_only true
EOF
        ollama create qwen3-embedding -f Modelfile
        echo "✅ Ollama model created from remote GGUF"
    fi
else
    echo "✅ Local GGUF model found: qwen3-embedding-0.6b.gguf"
    
    # Create optimized Modelfile for local GGUF
    cat > Modelfile << EOF
FROM ./qwen3-embedding-0.6b.gguf
PARAMETER num_ctx 8192
PARAMETER embedding_only true
EOF
    
    # Create the model in Ollama
    echo "🔧 Creating optimized Ollama model..."
    ollama create qwen3-embedding -f Modelfile
    echo "✅ Ollama model created successfully"
fi

# Step 2: Install Python dependencies
echo "📦 Step 2: Installing Python dependencies..."
pip3 install -r requirements.txt
echo "✅ Dependencies installed"

# Step 3: Setup Qdrant
echo "📦 Step 3: Setting up Qdrant vector database..."

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running! Please start Docker first."
    exit 1
fi

# Stop existing Qdrant container if it exists
if docker ps -a --format 'table {{.Names}}' | grep -q "qdrant"; then
    echo "Stopping existing Qdrant container..."
    docker stop qdrant || true
    docker rm qdrant || true
fi

# Start new Qdrant container
echo "Starting Qdrant container..."
docker run -d --name qdrant \
    -p 6333:6333 \
    -p 6334:6334 \
    -e QDRANT__SERVICE__API_KEY="your-super-secret-qdrant-api-key" \
    -v "$(pwd)/qdrant_storage:/qdrant/storage" \
    qdrant/qdrant

# Wait for Qdrant to be ready
echo "Waiting for Qdrant to start..."
for i in {1..30}; do
    if curl -s http://localhost:6333/health > /dev/null 2>&1; then
        echo "✅ Qdrant is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Qdrant failed to start"
        exit 1
    fi
    sleep 2
done

# Step 4: Start the API in background
echo "📦 Step 4: Starting OpenAI-compatible API..."

stop_api_process

if is_port_in_use $API_PORT; then
    echo "❌ Port $API_PORT still in use!"
    exit 1
fi

nohup python3 qwen3-api.py > "$LOG_FILE" 2>&1 &
API_PID=$!
echo $API_PID > "$PID_FILE"
echo "✅ API started (PID: $API_PID)"

echo "Waiting for API to start..."
for i in {1..20}; do
    if ! is_process_running "$API_PID"; then
        echo "❌ API died!"
        tail -20 "$LOG_FILE"
        rm -f "$PID_FILE"
        exit 1
    fi
    
    if curl -s http://localhost:$API_PORT/health > /dev/null 2>&1; then
        echo "✅ API ready"
        break
    fi
    
    if [ $i -eq 20 ]; then
        echo "❌ Timeout"
        tail -20 "$LOG_FILE"
        kill $API_PID 2>/dev/null || true
        rm -f "$PID_FILE"
        exit 1
    fi
    sleep 2
done

# Step 5: Setup Qdrant vector store
echo "📦 Step 5: Setting up Qdrant vector store..."
python3 qdrantsetup.py
echo "✅ Qdrant vector store configured"

# Step 6: Run verification tests
echo "📦 Step 6: Running verification tests..."
python3 test_qwen_features.py

# Summary
echo ""
echo "🎉 Setup complete! Your Qwen3 embedding system is ready for Roo Code."
echo ""
echo "🔧 Roo Code Configuration:"
echo "   Embeddings Provider: OpenAI-compatible"
echo "   Base URL: http://localhost:8000"
echo "   API Key: your-super-secret-qdrant-api-key"
echo "   Model: qwen3"
echo "   Embedding Dimension: 1024"
echo "   Qdrant URL: http://localhost:6333"
echo "   Qdrant API Key: your-super-secret-qdrant-api-key"
echo "   Collection Name: qwen3_embedding"
echo ""
echo "🚀 Services running:"
echo "   - Qwen3 API: http://localhost:8000"
echo "   - Qdrant: http://localhost:6333"
echo "   - Ollama: http://localhost:11434"
echo ""
echo "💡 Stop API: kill \$(cat $PID_FILE) or pkill -f qwen3-api.py"
echo "💡 View logs: tail -f $LOG_FILE"
echo "💡 Restart: ./setup.sh (safe - preserves data)"
