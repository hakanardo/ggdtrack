full_run_local:
	docker run -ti -v `pwd`:/workspace -v /home/hakan/src/duke:/home/hakan/src/duke ggdtrack python full_run.py
full_run:
	docker run -ti -v /home/hakan/src/duke/cachedir:/workspace/cachedir -v /home/hakan/src/duke:/home/hakan/src/duke ggdtrack python full_run.py
test:
	docker run -ti -v `pwd`:/workspace ggdtrack py.test -v
build:
	docker build -t ggdtrack .
flatten:
	docker run --name container-to-be-flatten ggdtrack
	export container-to-be-flatten | docker import - ggdtrack:flatt
	docker rm container-to-be-flatten
push:
	docker tag -f ggdtrack hakanardo/ggdtrack
	docker push hakanardo/ggdtrack

.PHONY: build full_run test
