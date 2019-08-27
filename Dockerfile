FROM tensorflow/tensorflow:latest-py3
RUN apt-get update
RUN apt-get install python3-lxml -y
RUN pip3 install -U scikit-learn 
RUN pip3 install -U Flask
RUN pip3 install -U pandas

COPY . /opt/mearin
WORKDIR = /opt/mearin

CMD cd /opt/mearin/ && python3 servico.py

EXPOSE 5000 5000