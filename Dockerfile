FROM mambaorg/micromamba@sha256:5e3b9c781f71c5d715a18216cdc9ce778bc974f478c6757a6e853a11d05fad12

WORKDIR /app

# Copy envornment file
COPY environment.yml .

USER root
# Create Conda environment with only conda packages
RUN micromamba env create --copy -p /env --file environment.yml && \
    micromamba clean --all --yes

USER $MAMBA_USER

# Make the dependencies accessible
ENV PATH=/env/bin:$PATH
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Copy the rest of the app
COPY --chmod=777 . .

EXPOSE 8000

RUN python -c "import whisper; whisper.load_model('turbo')"

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
