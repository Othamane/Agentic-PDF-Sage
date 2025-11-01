# 🤖 PDF Sage - Intelligent Document Chat System

**Production-ready RAG chatbot with enhanced AI reasoning - Powered by Gemini**

## 🎯 What's Included

✅ **Complete Backend** (FastAPI + PostgreSQL)
✅ **Complete Frontend** (React + Modern UI)
✅ **Gemini LLM Integration** (Fast & Reliable)
✅ **Enhanced Vector Search** (FAISS + Sentence Transformers)
✅ **Production Docker Setup**
✅ **Security & Monitoring**
✅ **Status Management & Debugging**

## 🚀 Quick Start (5 Minutes)

### 1. Prerequisites
- Docker & Docker Compose
- 8GB+ RAM (16GB recommended)
- 10GB+ free disk space
- Gemini API Key (free from Google AI Studio)

### 2. Clone & Setup
```bash
git clone <your-repository>
cd pdf-sage

# Copy environment template
cp .env.production.template .env.production
```

### 3. Configure Environment
Edit `.env.production` - **Add your Gemini API key for best performance**
```bash
# Required: Basic security
DB_PASSWORD=your_secure_password_123
SECRET_KEY=your-super-secret-key-change-this-in-production

# RECOMMENDED: Gemini API (fast, reliable, high-quality)
GEMINI_API_KEY=your_gemini_api_key_here
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash

# Alternative: Free local LLM (slower but no API costs)
# LLM_PROVIDER=ollama
# LLM_MODEL=llama2

# Optional: Production domain
ALLOWED_HOSTS='["yourdomain.com","localhost"]'
```

### 4. Get Gemini API Key (Free)
```bash
# Visit: https://aistudio.google.com/app/apikey
# Create free account
# Generate API key
# Add to .env.production as GEMINI_API_KEY
```

### 5. Launch Application
```bash
# Start all services
docker-compose up --build

# If using Ollama fallback: Wait for models to download (~5-10 minutes)
# Monitor progress: docker-compose logs -f ollama-init
```

### 6. Access Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/chat/status

## 📁 Enhanced Project Structure
```
pdf-sage/
├── backend/                           # FastAPI Backend
│   ├── app/
│   │   ├── api/v1/endpoints/         # Enhanced API Endpoints
│   │   │   ├── enhanced_chat_endpoints.py    # ✅ Chat with Gemini
│   │   │   ├── enhanced_document_endpoints.py # ✅ Document management
│   │   │   └── health.py                     # ✅ Health checks
│   │   ├── core/                    # Core Configuration
│   │   │   ├── config.py            # ✅ Enhanced settings
│   │   │   ├── database.py          # ✅ PostgreSQL + async
│   │   │   └── logging_config.py    # ✅ Structured logging
│   │   ├── models/                  # Database Models
│   │   │   ├── conversation.py      # ✅ Chat conversations
│   │   │   ├── document.py          # ✅ Document metadata
│   │   │   ├── agent_step.py        # ✅ Reasoning steps
│   │   │   └── retrieval_log.py     # ✅ Source tracking
│   │   ├── services/                # Enhanced Business Logic
│   │   │   ├── gemini_llm_service.py        # ✅ Gemini + HF fallback
│   │   │   ├── enhanced_agent_service.py    # ✅ Advanced reasoning
│   │   │   ├── enhanced_document_service.py # ✅ Status management
│   │   │   └── enhanced_vector_service.py   # ✅ Improved search
│   │   └── main.py                  # ✅ FastAPI app
│   ├── db/init.sql                  # ✅ Database schema
│   ├── requirements.txt             # ✅ Dependencies with Gemini
│   └── Dockerfile                   # ✅ Production container
├── frontend/                        # React Frontend
│   ├── src/
│   │   ├── components/              # React Components
│   │   │   ├── Chat/               # Enhanced Chat Interface
│   │   │   ├── Documents/          # Document Management
│   │   │   ├── Layout/             # App Layout
│   │   │   └── UI/                 # Reusable Components
│   │   ├── pages/                  # Page Components
│   │   ├── services/               # API Services
│   │   └── styles/                 # Modern CSS
│   ├── package.json                # ✅ Dependencies
│   └── Dockerfile                  # ✅ Production container
├── docker-compose.yml              # ✅ Full stack orchestration
├── .env.production.template        # ✅ Environment template
└── README.md                       # ✅ This complete guide
```

## 🎯 Enhanced Features

### 🤖 **Advanced AI Reasoning**
- ✅ **Gemini 2.5 Flash** - Latest Google AI model (fast, accurate)
- ✅ **Multi-step reasoning** (Planning → Retrieval → Synthesis → Validation)
- ✅ **Transparent reasoning traces**
- ✅ **Automatic fallback** to local models if needed
- ✅ **Enhanced error handling** & recovery

### 🔍 **Intelligent Document Processing**
- ✅ **PDF upload & text extraction**
- ✅ **Advanced chunking & embeddings**
- ✅ **Enhanced vector similarity search** (FAISS)
- ✅ **Source citation & tracking**
- ✅ **Status consistency** across page reloads

### 💬 **Modern Chat Interface**
- ✅ **Real-time reasoning display**
- ✅ **Source citations with relevance scores**
- ✅ **Document selection & management**
- ✅ **Conversation history**
- ✅ **Enhanced debugging** information

### 📊 **Analytics & Monitoring**
- ✅ **System health monitoring**
- ✅ **Document processing stats**
- ✅ **Performance metrics**
- ✅ **Enhanced error tracking**
- ✅ **Debug endpoints** for troubleshooting

### 🔒 **Production Security**
- ✅ **Rate limiting** (per IP)
- ✅ **Security headers** (CSP, HSTS, etc.)
- ✅ **Input validation & sanitization**
- ✅ **CORS protection**
- ✅ **Non-root containers**

### ⚡ **Performance Stack**
- ✅ **Gemini API** - 10x faster than local models
- ✅ **Enhanced vector search** - Better chunk retrieval
- ✅ **Async processing** - Non-blocking operations
- ✅ **Database optimization** - Connection pooling
- ✅ **Smart caching** - Reduced API calls

## 🛠️ Using PDF Sage

### 1. Upload Documents
```bash
# Access document manager
http://localhost:3000/documents

# Upload PDF files via web interface
# Monitor processing status in real-time
# Files are automatically processed and vectorized
```

### 2. Start Intelligent Conversations
```bash
# Access chat interface
http://localhost:3000/chat

# Select documents to chat with
# Ask complex questions
# See detailed reasoning traces
# Review source citations
```

### 3. Monitor System Health
```bash
# Check system status
curl http://localhost:8000/api/chat/status

# Debug document processing
http://localhost:8000/api/debug/document/{document_id}

# Test vector search
http://localhost:8000/api/debug/vector-search
```

## 🔧 Configuration Options

### **LLM Providers (in order of recommendation)**

1. **Gemini (Recommended)**
```bash
   LLM_PROVIDER=gemini
   LLM_MODEL=gemini-2.5-flash
   GEMINI_API_KEY=your_key_here
```
   - ✅ Fastest responses (2-5 seconds)
   - ✅ Highest quality reasoning
   - ✅ Most reliable
   - 💰 Generous free tier

2. **Ollama (Free Local)**
```bash
   LLM_PROVIDER=ollama
   LLM_MODEL=llama2
```
   - ✅ Completely free
   - ⚠️ Slower responses (15-60 seconds)
   - ⚠️ Requires more RAM

3. **HuggingFace (Fallback)**
```bash
   LLM_PROVIDER=huggingface
```
   - ✅ Free
   - ⚠️ Limited model options

### **System Scaling**
```bash
# For high-traffic production
RATE_LIMIT_REQUESTS=1000
DATABASE_POOL_SIZE=20
VECTOR_CACHE_SIZE=100

# For development
DEBUG=true
ENABLE_VECTOR_DEBUG=true
ENABLE_LLM_DEBUG=true
```

## 🎯 Expected Performance

| Configuration | Response Time | Quality | Cost |
|---------------|---------------|---------|------|
| **Gemini 2.5 Flash** | 2-5 seconds | Excellent | Free tier + usage |
| **Ollama Local** | 15-60 seconds | Good | Completely free |
| **HuggingFace** | 10-30 seconds | Fair | Free |

## 🆘 Troubleshooting

### Common Issues & Solutions

#### 1. **"No LLM providers available"**
- ✅ Check `GEMINI_API_KEY` in `.env.production`
- ✅ Verify API key at https://aistudio.google.com/app/apikey
- ✅ Check backend logs: `docker-compose logs backend`

#### 2. **"No chunks found in similarity search"**
- ✅ Ensure documents are fully processed (status: "processed")
- ✅ Check vector store: `http://localhost:8000/api/debug/document/{id}`
- ✅ Re-upload document if processing failed

#### 3. **Slow response times**
- ✅ Switch to Gemini: Set `LLM_PROVIDER=gemini`
- ✅ Check system resources
- ✅ Monitor logs for timeouts

### **Debug Endpoints**
- `/api/chat/status` - System health
- `/api/debug/document/{id}` - Document status
- `/api/debug/vector-search` - Test search

### **Getting Help**
- Check logs: `docker-compose logs -f`
- API documentation: http://localhost:8000/docs
- Health status: http://localhost:8000/api/chat/status

---

**PDF Sage** - Transform your PDFs into intelligent conversations powered by advanced AI reasoning.