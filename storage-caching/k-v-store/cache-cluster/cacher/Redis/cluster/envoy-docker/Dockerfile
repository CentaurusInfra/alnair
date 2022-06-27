FROM envoyproxy/envoy:v1.20-latest

RUN apt-get update -qq && apt-get install -yqq redis-tools curl dnsutils gettext-base
RUN mkdir -p /etc/envoy/
COPY entrypoint.sh /
RUN chmod +x /entrypoint.sh
EXPOSE 6379
ENTRYPOINT ["/entrypoint.sh"]