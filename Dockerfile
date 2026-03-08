FROM mambaorg/micromamba:git-c0f93d2

WORKDIR /app

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes

# Make the dependencies accessible
ENV PATH=/env/bin:$PATH
ARG MAMBA_DOCKERFILE_ACTIVATE=1

EXPOSE 8000

#RUN python -c "import whisper; whisper.load_model('turbo')"

# Command to run the application
COPY --chown=$MAMBA_USER:$MAMBA_USER main.py ./
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]