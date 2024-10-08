FROM automode_base

COPY custom-loopfunctions AutoMoDe-loopfunctions/loop-functions/custom-loopfunctions
# Add custom-loopfunctions to CMakeLists.txt
RUN cd AutoMoDe-loopfunctions \
&& echo "add_subdirectory(custom-loopfunctions)" >> loop-functions/CMakeLists.txt \
&& mkdir -p build && cd build \
&& cmake -DCMAKE_INSTALL_PREFIX=$ARGOS_INSTALL_PATH/argos3-dist -DCMAKE_BUILD_TYPE=Release .. \
&& make \
&& make install

COPY aac.argos /root/aac.argos