# set base image (host OS)
FROM python:3.7.7-slim
RUN apt-get update
# set the working directory in the container
WORKDIR /repo
# copy the dependencies file to the working directory
COPY requirements.txt /repo/requirements.txt
# install dependencies
RUN pip install --upgrade pip setuptools wheel && pip install -r requirements.txt
# copy the content of the local pwd directory to the container one
COPY . /repo
# command to run on container start
CMD ["bash"]