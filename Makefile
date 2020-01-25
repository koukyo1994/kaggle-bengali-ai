IMG := torch-catalyst
TAG := 2020.1.25

docker-build:
	make -C docker IMG=${IMG} TAG=${TAG}
