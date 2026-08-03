FROM mambaorg/micromamba:1.5.8

ENV PYTHONUNBUFFERED=1

# Build the conda environment defined by the project.
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba env create -y -f /tmp/environment.yml && micromamba clean --all --yes

WORKDIR /app
COPY --chown=$MAMBA_USER:$MAMBA_USER . /app

# Default command prints help; compose tasks override this.
ENTRYPOINT ["micromamba", "run", "-n", "p3.7torch1.8", "python"]
CMD ["predict.py", "--help"]