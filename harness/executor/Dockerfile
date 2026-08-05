FROM python:3.12-slim@sha256:423ed6ab25b1921a477529254bfeeabf5855151dc2c3141699a1bfc852199fbf

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.lock /tmp/requirements.lock

RUN pip install --no-cache-dir --no-deps -r /tmp/requirements.lock \
    && rm /tmp/requirements.lock

WORKDIR /workspace
