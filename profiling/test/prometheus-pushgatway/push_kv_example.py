import requests


def pushgateway_write():
    """the value prometheus store much be numerical, so no string is allow, no dictionanry"""
    job_name='my_custom_metrics1'

    instance_name='10.20.0.1:9000'

    team='alnair'

    payload_key='cpu_utilization'

    payload_value='21.90'


    pushgateway_addr = "10.145.83.40:9091" # enable host port, for outside cluster testing
    response = requests.post('http://10.145.83.40:9091/metrics/job/{j}/instance/{i}/team/{t}'.format(j=job_name, i=instance_name, t=team), data='{k} {v}\n'.format(k=payload_key, v=payload_value))
    # expect 200 response code
    print(response.status_code)
