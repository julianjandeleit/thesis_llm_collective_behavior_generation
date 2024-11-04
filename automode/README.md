# thesis_llm_collective_behavior_generation


first build base (automode with argos beta 48 and behavior trees controllers):
```
cd automode_base
docker build -t automode_base .
cd ..
```

then build actual automode image with our modifications:
```
docker build -t automode .
```

allow gui (optional, recommended) with "xhost +local:docker" or "xhost +local:podman"

test automode with:
```
docker run --rm -it --env="DISPLAY" --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --privileged automode AutoMoDe/bin/automode_main_bt -c aac.argos --bt-config --nroot 3 --nchildroot 1 --n0 0 --nchild0 2 --n00 6 --c00 5 --p00 0.26 --n01 5 --a01 0 --rwm01 5 --p01 0
```

mount a custom .argos file to `/root/aac.argos` to overwrite mission.

e.g.

```
 podman build -t automode . && podman run --rm -it --env="DISPLAY" --env="QT_X11_NO_MITSHM=1" -v ../ressources/examples/aggregation.argos:/root/aac.argos  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --privileged automode AutoMoDe/bin/automode_main_bt -c aac.argos --bt-config --nroot 3 --nchildroot 1 --n0 0 --nchild0 2 --n00 6 --c00 5 --p00 0.26 --n01 5 --a01 0 --rwm01 5 --p01 0 | sed '/STOP/d'
```