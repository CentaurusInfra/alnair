FROM python:3.8-alpine 
RUN mkdir app
WORKDIR /app
COPY manager ./
COPY grpctool ./grpctool
RUN pip install -r requirements.txt
RUN chmod +x *
CMD [ "python3", "manager.py" ]