# Create base image
FROM continuumio/miniconda3

# Set the working dir in the container
WORKDIR /Transparency

# Copy the content of the repo to workdir
COPY Transparency .

RUN apt-get -y install make \
      apt-get -y install g++

RUN /bin/bash -c 'conda create -n "maka_paper" python==3.7.9 -y \
      source activate maka_paper \
      conda env list \
      pip install -r Transparency/requirements.txt \
      python -m spacy download en \
      conda install -c anaconda jupyter -y \
      conda install -c anaconda pytest -y \
      export PYTHONPATH=$PYTHONPATH:$(pwd)'