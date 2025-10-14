# bchybrid
Enhance version of BCCL with the objectives of hybrid

### How to compile grpc proto file using python

```python
### Install grpc python tool
$ pip install grpcio grpcio-tools protobuf
```

```python
### Compile grpc at the location of *.proto file
$ python -m grpc_tools.protoc -I. --python_out=. --pyi_out=. --grpc_python_out=. gossip.proto
```

### How build image in MacOS for amd64 linux version using Docker command
```python
# $ docker buildx build --platform linux/amd64 -t <your-registry-username>/<image-name>:<tag> --push .
# For x86_64, x64

$ docker buildx build --platform linux/amd64 -t wwiras/simsba:v1 --push .
```

### How to remove all unused images in MacOS/Linux Docker command
```python
$ docker image prune -a
```

### Creating Pods based on image name and tag 
```python
$ helm install simcn ./chartsim --set testType=default,totalNodes=25,image.tag=v1,image.name=wwiras/simsba --debug
```

### Uninstall Pods  
```python
$ helm uninstall simcn
```