FROM ubuntu:18.04 as base

RUN apt-get update && apt-get install -y \
        vim \
        curl \
        python3 \
        python3-pip
RUN python3 -m pip install --upgrade pip
RUN pip3 install prometheus-api-client kubernetes statsmodels sklearn
# notebook for debug purpose
RUN pip3 install jupyter matplotlib
RUN pip3 install jupyter_http_over_ws ipykernel nbformat
RUN jupyter serverextension enable --py jupyter_http_over_ws

RUN mkdir /app
WORKDIR /app
EXPOSE 9999
COPY ./app.py .
RUN python3 -m ipykernel.kernelspec
#CMD ["bash", "-c", "jupyter notebook --notebook-dir=/app --ip 0.0.0.0 --port 9999 --no-browser --allow-root"]
CMD ["bash", "-c", "python3 app.py"]
