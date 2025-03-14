# Based on https://github.com/ContinuumIO/docker-images/commit/068122304e2f3a5072bd8a489fda343251a99e71
FROM debian:buster-slim

# AWS section based on https://vsupalov.com/docker-build-pass-environment-variables/
# $ docker build --build-arg var_name=${VARIABLE_NAME} (...)
# OR
# $ docker build --build-arg var_name (...)
ARG aws_access_key
ARG aws_secret_access
ARG s3_bkt

ENV AWS_ACCESS_KEY_ID=$aws_access_key
ENV AWS_SECRET_ACCESS_KEY=$aws_secret_access
ENV AWS_REGION=us-east-1
ENV S3_BUCKET=$s3_bkt

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENV PATH /opt/conda/bin:$PATH

COPY install-packages.sh .
RUN ./install-packages.sh

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN conda update conda --yes

RUN conda config --add channels conda-forge

RUN conda install \
    python=3.6.9 \
    awscli \
    boto3 \
    dask \
    python-dotenv \
    h5py \
    keras \
    maya \
    numpy \
    pandas \
    tensorflow'<2.0.0' \
    toml \
    pip \
    setuptools \
    wheel

RUN conda clean -yt
