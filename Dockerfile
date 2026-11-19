FROM python:3.11-slim
WORKDIR /srv
ENV PYTHONUNBUFFERED=1 HF_HOME=/srv/.hf_cache

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY data/sample_docs ./data/sample_docs
COPY eval ./eval

# Optional: bake models into the image so cold starts don't download them
# RUN python -c "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/all-MiniLM-L6-v2')"

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
