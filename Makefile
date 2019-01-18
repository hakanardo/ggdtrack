full_run:
	docker run -ti -v `pwd`:/workspace -v /home/hakan/src/duke:/home/hakan/src/duke ggdtrack python full_run.py
test:
	docker run -ti -v `pwd`:/workspace ggdtrack py.test -v
build:
	docker build -t ggdtrack .
flatten:
	docker run --name container-to-be-flatten ggdtrack
	export container-to-be-flatten | docker import - ggdtrack:flatt
	docker rm container-to-be-flatten

.PHONY: build full_run test
