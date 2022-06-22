

if __name__=="__main__":
    config = "backup-dir=/data/ flush_frequency=10 flush_amount=10"
    result = {}
    for item in map(lambda x: x.split("="), config.split()):
        result[item[0]] = item[1]
    print(result)