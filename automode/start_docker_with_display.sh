xhost +local:docker
# (or localhost +local:podman respectively)
docker run --rm -it \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    argos_docker


# maybe also requires --privileged flag