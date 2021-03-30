all:
	CC=gcc python3 setup.py build_ext --inplace

clean:
	rm *.c *.so