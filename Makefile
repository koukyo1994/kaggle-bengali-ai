IMG := torch-catalyst
TAG := 2020.1.25
NAME := bengali
USER := kaggle
CONFIG := config/se_resnext.bce.0.yml

docker-build:
	make -C docker/ IMG=${IMG} TAG=${TAG}

env:
	docker run --rm -it --init \
	--ipc host \
	--name ${NAME} \
	--volume `pwd`:/app/${NAME} \
	-w /app/${NAME} \
	--user `id -u`:`id -g` \
	--publish 9000:9000 \
	${IMG}:${TAG} /bin/bash

jupyter:
	sudo chown ${USER}:${USER} /home/user/.jupyter
	jupyter lab --port 9000 --ip 0.0.0.0 --NotebookApp.token=

prepare:
	python prepare.py



train:
	sudo chmod 777 /home/user
	mkdir -p /home/user/.cache
	sudo chmod 777 /home/user/.cache
	python train.py --config ${CONFIG}
