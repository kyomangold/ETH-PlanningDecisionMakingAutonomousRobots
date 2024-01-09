FROM python:3.8

## preliminary needed only for being compatible with M1 architecture
RUN apt update && apt upgrade -y
RUN apt install libgeos-dev python3-gdal libgdal-dev libspatialindex-dev libspatialindex-c6 -y

##install dependencies
WORKDIR /pdm4ar

RUN apt update
RUN apt upgrade -y
RUN apt install libgeos-dev -y
RUN apt install python3-gdal -y
RUN apt install libgdal-dev -y
RUN apt install libspatialindex-dev libspatialindex-c6 -y

COPY requirements.txt .
RUN pip install -r requirements.txt

# Install necessary system packages
RUN apt update && apt install -y cmake

# Install Python packages
RUN pip install cvxpy

COPY . .
RUN find .
ENV DISABLE_CONTRACTS=1

RUN python setup.py develop --no-deps


CMD ["bash"]
