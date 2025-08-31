# Ollama Setup Guide - Free LLM for Agentic PDF Sage

This guide explains how to set up **Ollama**, a completely free and open-source LLM solution that runs locally without any API costs.

## 🎯 Why Ollama?

- ✅ **100% Free** - No API costs or usage limits
- ✅ **Private** - All processing happens locally
- ✅ **Fast** - Direct local inference
- ✅ **Multiple Models** - Llama2, Mistral, CodeLlama, and more
- ✅ **Easy Setup** - Simple Docker deployment

## 🚀 Quick Start

### Option 1: Using Docker Compose (Recommended)

The provided `docker-compose.yml` includes Ollama automatically:

```bash
# Start all services including Ollama
docker-compose up --build

# Wait for models to download (first time only)
# This may take 5-10 minutes depending on your internet speed
```

### Option 2: Manual Ollama Installation

If you prefer to run Ollama separately:

```bash
# Install Ollama (Linux/Mac)
curl -fsSL https://ollama.ai/install.sh | sh

# Or download from https://ollama.ai for Windows

# Start Ollama server
ollama serve

# Pull models
ollama pull llama2        # ~3.8GB
ollama pull mistral       # ~4.1GB
ollama pull codellama     # ~3.8GB
```

## 🔧 Configuration

### Environment Variables

```bash
# LLM Provider Settings
LLM_PROVIDER=ollama
LLM_MODEL=llama2          # or mistral, codellama, etc.
OLLAMA_HOST=http://localhost:11434

# Embedding Settings (free)
EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### Available Models

| Model | Size | Description | Best For |
|-------|------|-------------|----------|
| `llama2` | 3.8GB | Meta's Llama 2 7B | General conversations |
| `mistral` | 4.1GB | Mistral 7B | Instruction following |
| `codellama` | 3.8GB | Code-specialized Llama | Code generation |
| `neural-chat` | 4.1GB | Intel's chat model | Conversations |
| `starling-lm` | 4.1GB | Berkeley's model | Reasoning tasks |

### Performance Recommendations

**Minimum Requirements:**
- 8GB RAM
- 4GB available disk space
- 2+ CPU cores

**Recommended:**
- 16GB RAM
- 8GB available disk space
- 4+ CPU cores
- SSD storage

## 🏃‍♂️ Running the Application

### Step 1: Start Services

```bash
# Clone the project
git clone <your-repo>
cd agentic-pdf-sage

# Copy environment file
cp .env.production.template .env.production

# Edit the environment file - NO API KEYS NEEDED!
nano .env.production
```

### Step 2: Configure Environment

```bash
# .env.production - Minimal free setup
ENVIRONMENT=production
DB_PASSWORD=your_secure_password
SECRET_KEY=your_secret_key

# Free LLM settings
LLM_PROVIDER=ollama
LLM_MODEL=llama2
OLLAMA_HOST=http://ollama:11434

# Free embeddings
EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### Step 3: Launch Application

```bash
# Start all services
docker-compose up --build

# First-time model download (wait patiently)
# Check logs: docker-compose logs -f ollama-init
```

### Step 4: Access Application

- **Frontend:** http://localhost:3000
- **Backend:** http://localhost:8000
- **Ollama API:** http://localhost:11434

## 🔍 Troubleshooting

### Common Issues

#### 1. Ollama Not Responding
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama service
docker-compose restart ollama
```

#### 2. Model Download Fails
```bash
# Check available space
df -h

# Manually pull model
docker-compose exec ollama ollama pull llama2
```

#### 3. Out of Memory
```bash
# Use smaller model
LLM_MODEL=llama2:7b-chat-q4_0  # Quantized version

# Or increase Docker memory limit
# Docker Desktop -> Settings -> Resources -> Memory
```

#### 4. Slow Performance
```bash
# Use quantized models for better performance
ollama pull llama2:7b-chat-q4_0    # 4-bit quantization
ollama pull mistral:7b-instruct-q4_0

# Update environment
LLM_MODEL=llama2:7b-chat-q4_0
```

### Performance Optimization

#### CPU Optimization
```bash
# Set CPU threads (in .env.production)
OLLAMA_NUM_THREAD=4  # Adjust based on your CPU
```

#### Memory Management
```bash
# Configure Docker memory limits
docker-compose.yml:
  ollama:
    deploy:
      resources:
        limits:
          memory: 6G  # Adjust based on available RAM
```

## 🔄 Alternative Free LLMs

If Ollama doesn't work for your setup, the application automatically falls back to HuggingFace transformers:

### HuggingFace Setup
```bash
# Environment configuration
LLM_PROVIDER=huggingface
LLM_MODEL=microsoft/DialoGPT-medium

# Optional: HuggingFace token for private models
HUGGINGFACE_TOKEN=your_hf_token_here
```

### Available HuggingFace Models
- `microsoft/DialoGPT-medium` - Conversational AI
- `google/flan-t5-base` - Instruction following
- `bigscience/bloom-560m` - Multilingual model

## 🧪 Testing Your Setup

### 1. Test Ollama Directly
```bash
# Test model availability
curl http://localhost:11434/api/tags

# Test generation
curl http://localhost:11434/api/generate -d '{
  "model": "llama2",
  "prompt": "Why is the sky blue?",
  "stream": false
}'
```

### 2. Test Backend Integration
```bash
# Check backend health
curl http://localhost:8000/health

# Test chat endpoint (once implemented)
curl -X POST http://localhost:8000/api/v1/chat/send \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, how are you?"}'
```

## 📊 Performance Comparison

| Model | Size | Speed | Quality | Memory Usage |
|-------|------|-------|---------|--------------|
| Llama2 | 3.8GB | Medium | High | 4-6GB |
| Mistral | 4.1GB | Fast | High | 4-6GB |
| CodeLlama | 3.8GB | Medium | High (Code) | 4-6GB |
| HF DialoGPT | 800MB | Fast | Medium | 1-2GB |

## 🔧 Advanced Configuration

### Custom Model Configuration
```bash
# Create custom Modelfile
cat > Modelfile << EOF
FROM llama2
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER top_k 40
SYSTEM You are a helpful AI assistant specialized in document analysis.
EOF

# Create custom model
ollama create pdf-sage -f Modelfile
```

### Production Scaling
```bash
# Multiple Ollama instances for load balancing
# Use nginx upstream configuration
# Deploy on multiple servers with shared storage
```

## 🎉 Success!

Once everything is running:

1. ✅ Ollama serves local LLM (no API costs)
2. ✅ Sentence-transformers handles embeddings (free)
3. ✅ PostgreSQL stores conversation data
4. ✅ React frontend provides chat interface
5. ✅ FastAPI backend orchestrates everything

**Total cost: $0** 🎉

## 📚 Resources

- [Ollama Documentation](https://ollama.ai/docs)
- [Available Models](https://ollama.ai/library)
- [HuggingFace Models](https://huggingface.co/models)
- [Sentence Transformers](https://www.sbert.net/)

## 🤝 Support

If you encounter issues:

1. Check the logs: `docker-compose logs -f`
2. Verify system requirements
3. Try a smaller model first
4. Open an issue on GitHub

---

**Enjoy your free, private AI assistant!** 🚀