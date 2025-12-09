# syntax=docker/dockerfile:1

# Visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

FROM python:3.10.11

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1 

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1
ENV MPLCONFIGDIR=/tmp/matplotlib

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

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a bind mount to requirements.txt to avoid copying them into into this layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

# Switch to the non-privileged user to run the application.
USER appuser

# Copy the source code into the container.
COPY ./peregrin_app .

# Expose the port that the application listens on.
EXPOSE 55157

# Command for running the application.
CMD ["shiny", "run", "--port", "55157", "app.py"]