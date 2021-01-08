# source for jupyter: https://u.group/thinking/how-to-put-jupyter-notebooks-in-a-dockerfile/
# base image (host OS) setting
FROM python:3.8-slim-buster
RUN apt-get update
# working directory setting
RUN mkdir repo
WORKDIR /repo
COPY . .
# main dependencies installation and personal modules testing
RUN pip install --upgrade pip setuptools wheel && pip install -r requirements.txt
RUN pip install jupyter
RUN py.test
# Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]
# command that starts up the notebook at the end of the dockerfile
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
