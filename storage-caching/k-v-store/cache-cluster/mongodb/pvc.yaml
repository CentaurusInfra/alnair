kind: PersistentVolumeClaim
apiVersion: v1
metadata:
  name: mongodb-pvc
spec:
  storageClassName: mongodb-storageclass
  accessModes:
    - ReadWriteOnce
  volumeMode: Filesystem
  resources:
    requests:
      storage: 1Gi