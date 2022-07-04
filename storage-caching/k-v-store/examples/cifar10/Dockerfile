FROM pytorch/pytorch:latest
RUN mkdir app
WORKDIR /app
COPY lib ./lib
COPY cifar10/src ./
RUN pip install -r requirements.txt
RUN chmod +x *
CMD [ "python3", "main.py" ]