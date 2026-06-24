FROM rocker/r-ver:4.4.0

# Install system dependencies for plumber, ranger, and xgboost
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    make \
    zlib1g-dev \
    libsodium-dev \
    libfontconfig1-dev \
    libfreetype6-dev \
    libpng-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    pkg-config \
    cmake \
    libuv1-dev \
    libtiff5-dev \
    libjpeg-dev \
    libwebp-dev \
    && rm -rf /var/lib/apt/lists/*

ENV RENV_CONFIG_BIOCONDUCTOR_ENABLED=false

WORKDIR /app

# Restore R packages using renv.lock directly
COPY R_project/renv.lock ./R_project/
RUN Rscript -e "install.packages('renv', repos='https://cloud.r-project.org')"
RUN Rscript -e "renv::restore(lockfile='R_project/renv.lock', prompt=FALSE)"

# Copy code and models
COPY R_project/api.R R_project/
COPY R_project/R/ ./R_project/R/
COPY R_project/models/production_model_r.rds ./R_project/models/
COPY data/ ./data/

WORKDIR /app/R_project

EXPOSE 8001

CMD ["Rscript", "-e", "pr <- plumber::pr('api.R'); plumber::pr_run(pr, host='0.0.0.0', port=8001)"]
