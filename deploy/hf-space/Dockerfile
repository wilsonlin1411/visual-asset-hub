# Online demo on a Hugging Face Docker Space (CPU Basic: 2 vCPU, 16 GB RAM).
# The Space repo holds only this file and README.md: code comes from GitHub at tag REF, models
# from the GitHub Release. To deploy a new version, push a new tag and change REF.
ARG REPO=https://github.com/wilsonlin1411/visual-asset-hub.git
ARG REF=v1.1.0
ARG MODELS_URL=https://github.com/wilsonlin1411/visual-asset-hub/releases/download/v1.0.0/vah-models.zip

FROM node:22-bookworm-slim AS web
ARG REPO
ARG REF
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch "$REF" "$REPO" /src && rm -rf /src/.git
WORKDIR /src/frontend
RUN npm ci && npm run build && rm -rf node_modules

FROM python:3.11-slim
ARG MODELS_URL
# libgl1 / libglib2.0-0: required by opencv-contrib-python
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl unzip \
    && rm -rf /var/lib/apt/lists/*
# Spaces run the container as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user --from=web /src/requirements.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt
RUN curl -fsSL -o /tmp/models.zip "$MODELS_URL" && unzip -q /tmp/models.zip -d . && rm /tmp/models.zip
COPY --chown=user --from=web /src/ ./

# Demo library built into the image from the 10 public test photos: every visitor gets a copy,
# and a restart (the Space disk is not persistent) brings it back to this state.
ENV VAH_DEMO=1 VAH_DATA_DIR=/home/user/app/demo-data VAH_SESSIONS_DIR=/home/user/app/sessions
RUN python -m backend.cli.library ingest tests/images \
    && python tools/seed_library.py \
    && python -m backend.cli.library thumbs

EXPOSE 7860
CMD ["python", "-m", "backend.api", "--host", "0.0.0.0", "--port", "7860"]
