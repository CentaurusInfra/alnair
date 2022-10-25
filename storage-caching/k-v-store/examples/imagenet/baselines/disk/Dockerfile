FROM pytorch/pytorch:latest
RUN apt update && apt install vmtouch
RUN mkdir app
WORKDIR /app
COPY main.py ./
COPY folder.py ./
COPY requirements.txt ./
RUN pip install -r requirements.txt
RUN chmod +x *
RUN cp folder.py /opt/conda/lib/python3.8/site-packages/torchvision/datasets/
# CMD [ "python3", "main.py" ]