FROM pytorch/pytorch:latest
RUN mkdir app
WORKDIR /app
COPY main.py ./
COPY requirements.txt ./
RUN pip install -r requirements.txt
RUN chmod +x *
CMD [ "python3", "main.py" ]