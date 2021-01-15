FROM mindspore-builder:1.1.0

WORKDIR /src
ADD . .
RUN ./prebuild.sh
RUN ./build.sh
RUN python -m pip install -U mindspore/output/*.whl
RUN cd ./mindspore/KungFu && go install -v ./srcs/go/cmd/kungfu-run
