FROM mambaorg/micromamba:git-c0f93d2

WORKDIR /app

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes

# Activate Conda Environment
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Install the used model
RUN python -c "import whisper; whisper.load_model('turbo')"

EXPOSE 8000

COPY --chown=$MAMBA_USER:$MAMBA_USER main.py ./
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
