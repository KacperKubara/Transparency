# Create base image
FROM continuumio/miniconda3 AS system_build

# Set the working dir in the container
WORKDIR /Transparency

# Copy the content of the repo to workdir
COPY Transparency .

RUN apt-get -y install make
RUN apt-get -y install g++

FROM system_build
RUN conda create -n "maka_paper" python==3.7.9 -y

# Override default shell command
SHELL ["conda", "run", "-n", "maka_paper", "/bin/bash", "-c"]

# Finish installation on the conda env
RUN conda env list 
RUN pip install -r requirements.txt 
RUN python -m spacy download en 
RUN conda install -c anaconda jupyter -y 
RUN conda install -c anaconda pytest -y 
RUN export PYTHONPATH=$PYTHONPATH:$(pwd)