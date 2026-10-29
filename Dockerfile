FROM mambaorg/micromamba:1.5.8

ENV PYTHONUNBUFFERED=1

# Build the conda environment defined by the project.
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba env create -y -f /tmp/environment.yml && micromamba clean --all --yes

WORKDIR /app
COPY --chown=$MAMBA_USER:$MAMBA_USER . /app

# NiChart bind-mounts the output directory from shared storage (FSx on cloud),
# where it is owned by root. The micromamba base image defaults to
# USER mambauser (uid 57439), which cannot write there, so run as root. The
# local DockerBackend passes an explicit --user, which still overrides this.
USER root

# micromamba writes a proc dir under $HOME. NiChart's local backend runs this
# image with --user <host uid>:<gid>, a uid with no passwd entry, so HOME would
# default to / and micromamba would abort before reaching the wrapper. /tmp is
# world-writable, so this works under root, mambauser and an arbitrary uid alike.
ENV HOME=/tmp

# Default command prints help; compose tasks override this.
ENTRYPOINT ["micromamba", "run", "-n", "p3.7torch1.8", "python"]
CMD ["predict.py", "--help"]