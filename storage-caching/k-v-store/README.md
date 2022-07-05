# K-V-Store

## Set up the Cache Cluster
Redis configurations are saved in [config.yaml](/cache-cluster/cacher/Redis/cluster/config.yaml). Please change and remember the `masterauth` and `requirepass` fields and uncomment the `bind=0.0.0.0`. For data eviction, consider to configure `maxmemory-policy`, `maxmemory` and `maxmemory-samples`. To run more instances, change the `replicas` field in the [cluster.yaml]((/cache-cluster/cacher/Redis/cluster/cluster.yaml)) file.

### Step 1. Set up the Redis Cluster with a server Statefulset (3 replicas).

Edit the [secret.yaml](cache-cluster/cacher/Redis/cluster/secret.yaml) to set passwords. Then execute:

```bash
cd cache-cluster/cacher/Redis/cluster
sh setup.sh init # enter `yes` when you see the prompt
```

If clients are from a different K8s cluster, you need to enable the envoy-proxy mode. To do so, execute:
```bash
kubectl apply -f envoy-proxy-config.yaml
kubectl apply -f envoy-proxy-deploy.yaml
```
Envoy-proxy service is exposed through NodePort with targetPort 30001. To get the node IP envoy-proxy is using:
```bash
kubectl get pods -o wide | grep envoy-proxy | awk '{ print $7 }'
```

### Step 2. Set up a HA MongoDB cluster using Statefulset
```bash
# please ensure you create diectory /mnt/data on all nodes
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

### Step 3. Deploy Global Manager (GM) Deployment
Edit the [configmap.yaml](cache-cluster/manager/configmap.yaml) to define configurations of the manager, mongodb, and redis_proxy. To enable data persistence, execute [nfs.sh](cache-cluster/manager/nfs.sh to set up NFS server on all worker nodes. Then execute:
```bash
cd cache-cluster/manager/
kubectl apply -f .
```