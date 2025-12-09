# syntax=docker/dockerfile:1

# Visit the Dockerfile reference guide at https://docs.docker.com/go/dockerfile-reference/

FROM python:3.10.11

# Prevents Python from writing pyc files, buffering stdout and stderr to avoid crashes without emitting any logs due to buffering.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /peregrin_app

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser 

# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid copying them into this layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt


USER appuser

COPY ./peregrin_app .

EXPOSE 56269

CMD ["shiny", "run", "--port", "56269", "app.py"]