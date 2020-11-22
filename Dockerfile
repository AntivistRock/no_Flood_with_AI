FROM nvidia/cuda
WORKDIR /app
RUN apt-get update
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
COPY requirements.txt ./
RUN pip3 install -r requirements.txt
COPY . .
ENTRYPOINT ["python3", "forecast.py"]
