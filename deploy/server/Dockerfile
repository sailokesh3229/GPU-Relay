# GPU Relay website: accounts for many users, each with their own GPU Relay (`gpurelay gateway`).
# Built by Render from the repo root (see render.yaml); works on any container host.
#
#   docker build -f deploy/server/Dockerfile -t gpurelay-site .
#   docker run -p 7860:7860 -e DATABASE_URL -e GPURELAY_SESSION_KEY -e GPURELAY_VAULT_KEY \
#       gpurelay-site
#
# Settings and secrets come from the host's environment settings, never from files
# (docs/multi-user.md lists them). Users' platform tokens are kept sealed in the database.
# The disk may be wiped on every restart: each user's GPU Relay restores its job database
# from that user's own storage repo and backs it up there. The single-owner server of earlier
# versions is still `gpurelay serve`.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

RUN useradd --create-home --uid 1000 relay
ENV HOME=/home/relay \
    GPURELAY_HOME=/home/relay/.gpurelay \
    GPURELAY_USERS_DIR=/home/relay/users \
    PATH=/home/relay/.local/bin:$PATH

WORKDIR /home/relay/app
COPY --chown=relay pyproject.toml README.md ./
COPY --chown=relay src ./src
RUN pip install ".[kaggle,dashboard,modal,server]"

USER relay
EXPOSE 7860
CMD ["gpurelay", "gateway"]
