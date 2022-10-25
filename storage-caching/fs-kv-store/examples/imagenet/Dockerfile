FROM pytorch/pytorch:latest
RUN mkdir app
WORKDIR /app
COPY imagenet/src ./
RUN pip install -r requirements.txt
RUN chmod +x *
CMD [ "python3", "main.py", "-j", "8", "--pretrained", "-e"]