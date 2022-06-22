# K-V-Store

## Set up the Cache Cluster
Redis configurations are saved in [redis-config.yaml](/cache-cluster/cacher/Redis/cluster/redis-config.yaml). Please change and remember the `masterauth` and `requirepass` fields and uncomment the `bind=0.0.0.0`. For data eviction, consider to configure `maxmemory-policy`, `maxmemory` and `maxmemory-samples`. To run more instances, change the `replicas` field in the [redis-cluster.yaml]((/cache-cluster/cacher/Redis/cluster/redis-cluster.yaml)) file.

### Step 1. Set up the Redis Cluster with a server Statefulset (3 replicas) and a proxy Deployment (2 replicas).
```bash
cd cache-cluster/cacher/Redis/cluster
sh setup.sh init # enter `yes` when you see the prompt

# get nodes where redis-proxy pods are running
# the service is exposed through NodePort with targetPort 30001
kubectl get pods -o wide | grep redis-proxy | awk '{ print $7 }'
```

### Step 2. Set up a HA MongoDB cluster using Statefulset
```bash
# create diectory /mnt/data on all nodes
kubectl config set-context --current --namespace=default
cd cache-cluster/cacher/mongodb
sh setup.sh init

# execute the follwing commands in the mongodb console:
rs.initiate()
var cfg = rs.conf()
cfg.members[0].host="mongo-0.mongo:27017"
rs.reconfig(cfg)
rs.add("mongo-1.mongo:27017")
rs.status()
db.createUser(
{
    user: "alnair",
    pwd: "alnair",
    roles: [
            { role: "userAdminAnyDatabase", db: "admin" },
            { role: "readWriteAnyDatabase", db: "admin" },
            { role: "dbAdminAnyDatabase", db: "admin" },
            { role: "clusterAdmin", db: "admin" }
        ]
})
exit()

# get the Cluser-IP and targetPort of mongodb service
kubectl get svc mongo
```

Step 3. Prepare and set up the Global Manager (GM) Deployment

```bash
# create GM image
cd src
docker build -t alnairpod:manager -f manager/Dockerfile .
docker tag alnairpod:manager alnair/alnairpod:manager
docker push alnair/alnairpod:manager

# start the GM deployment
kubectl apply -f manager/configmap.yaml
kubectl apply -f manager/deployment.yaml
```