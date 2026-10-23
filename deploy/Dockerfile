# GPU Relay controller for an always-on CPU machine (plan Phase 9 item 6).
#
#   docker build -f deploy/Dockerfile -t gpurelay .
#   docker run -d --name gpurelay --restart unless-stopped -v gpurelay-data:/data \
#       -e HF_TOKEN -e KAGGLE_API_TOKEN gpurelay up
#
# Tokens come from the environment (no OS keyring in a container). State, config, and the
# controller lock live in the /data volume; jobs and checkpoints live in your storage repo.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GPURELAY_HOME=/data

RUN useradd --create-home --uid 1000 relay \
    && mkdir -p /data && chown relay:relay /data

WORKDIR /opt/gpurelay
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[kaggle,dashboard]"

USER relay
VOLUME ["/data"]
EXPOSE 8765
ENTRYPOINT ["gpurelay"]
CMD ["up"]
