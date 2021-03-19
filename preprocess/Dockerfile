# Use an official Python runtime as a parent image
FROM continuumio/miniconda3

WORKDIR /hydromet

RUN apt-get update && apt-get install -y gdb

COPY environment.yml /hydromet
RUN conda env create -f /hydromet/environment.yml
ENV PATH /opt/conda/envs/hydrometenv/bin:$PATH
RUN /bin/bash -c "source activate hydrometenv"

COPY src /hydromet/src
