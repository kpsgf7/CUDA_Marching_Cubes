all: build

build: kpsgf7_marching_cubes

kpsgf7_marching_cubes.o:kpsgf7_marching_cubes.cu
	nvcc -ccbin g++ -I../../common/inc -m64 -o kpsgf7_marching_cubes.o -c kpsgf7_marching_cubes.cu

kpsgf7_marching_cubes: kpsgf7_marching_cubes.o
	nvcc -ccbin g++ -m64 -o kpsgf7_marching_cubes marching_cubes.o

clean:
	rm -f kpsgf7_marching_cubes kpsgf7_marching_cubes.o   output.ply

clobber: clean
