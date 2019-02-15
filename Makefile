full_run_local:
	nvidia-docker run --rm -ti -v `pwd`:/workspace ggdtrack python full_run.py
full_run:
	nvidia-docker run --rm -ti -v /home/hakan/src/duke/cachedir:/workspace/cachedir -v /home/hakan/src/duke:/workspace/data hakanardo/ggdtrack:dev python full_run.py
test:
	docker run --rm -ti -v `pwd`:/workspace ggdtrack py.test -v
build:
	docker build -t ggdtrack .
flatten:
	docker run --name container-to-be-flatten ggdtrack
	export container-to-be-flatten | docker import - ggdtrack:flatt
	docker rm container-to-be-flatten
push:
	docker tag -f ggdtrack hakanardo/ggdtrack:dev
	docker push hakanardo/ggdtrack:dev

.PHONY: build full_run test
