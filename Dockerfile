ARG BASE_IMAGE=dockerpull.org/kenneth850511/llamafactory:v0.9.1
FROM ${BASE_IMAGE}

# Define installation arguments
ARG PIP_INDEX=https://pypi.org/simple

# Set the working directory
WORKDIR /workspace
ENV PYTHONPATH /workspace/

# Install the requirements
COPY requirements.txt /workspace
RUN pip config set global.index-url "$PIP_INDEX" && \
    pip config set global.extra-index-url "$PIP_INDEX" && \
    python -m pip install -r requirements.txt

# Copy the rest of the application into the image
COPY . /workspace
RUN python -m pip install -e .
