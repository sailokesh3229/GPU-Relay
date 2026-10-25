# GPU Relay server: the controller and the web app behind a login (`gpurelay serve`).
# Built by Render from the repo root (see render.yaml); works on any container host.
#
#   docker build -f deploy/server/Dockerfile -t gpurelay-server .
#   docker run -p 7860:7860 -e GPURELAY_HF_REPO -e HF_TOKEN -e KAGGLE_API_TOKEN \
#       -e GPURELAY_OWNER_PASSWORD -e GPURELAY_VIEWER_PASSWORD gpurelay-server
#
# Secrets come from the host's environment settings, never from files. Optional: Modal, with
# GPURELAY_MODAL=1 (your confirmation of Modal's terms), MODAL_TOKEN_ID and MODAL_TOKEN_SECRET.
# The disk may be wiped on every restart: `gpurelay serve` restores the job database from the
# private storage repo and backs it up there every 10 minutes and on shutdown.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

RUN useradd --create-home --uid 1000 relay
ENV HOME=/home/relay \
    GPURELAY_HOME=/home/relay/.gpurelay \
    PATH=/home/relay/.local/bin:$PATH

WORKDIR /home/relay/app
COPY --chown=relay pyproject.toml README.md ./
COPY --chown=relay src ./src
RUN pip install ".[kaggle,dashboard,modal]"

USER relay
EXPOSE 7860
CMD ["gpurelay", "serve"]
