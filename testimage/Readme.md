use this docker container to determine if docker gui is working on your setup as described. If not, the setup is different than expected and other methods may need to be used in order to use docker with gui mode.

Tested on Ubuntu 24.04 under x11.

Build docker image
```
docker build -t testimage .
```

Allow docker to use x11
```
xhost +local:docker
```

Run xeyes test with gui parameters on docker launch. Successful if you see a window with two eyes.

```
docker run --rm -it --env="DISPLAY" --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" testimage xeyes
```