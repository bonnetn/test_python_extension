.PHONY: clean
clean:
	rm -rf build/

.PHONY: build
build:
	python3 setup.py build

.PHONY: install
install:
	python3 setup.py install

.PHONY: single_run
single_run:
	python3 single_run.py

.PHONY: benchmark
benchmark:
	python3 test.py
