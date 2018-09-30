CXX=g++
CXXFLAGS=-O3 -Wall -Wextra -std=c++11 -g
LDLIBS=-lm -fopenmp
OBJDIR=obj

NVCC=nvcc
CUFLAGS=-O3 -std=c++11 -g
CULDLIBS= -lcusolver -lm -lcudart -lcuda
CULINK=-L/usr/local/cuda/lib64
CUOBJDIR=cuobj

BINDIR=bin

vpath %.cpp source
vpath %.hpp source source/cuda
vpath %.cu source/cuda

OBJECTS=$(addprefix $(OBJDIR)/, \
          demo.o)

CUOBJECTS=$(addprefix $(CUOBJDIR)/, \
        anlm.o \
        DMat.o \
        init.o )

TESTS=test_blonde128 \
	  test_lena64 \
	  test_lena128 \
	  test_lena192 \
	  test_livingroom64 \
	  test_livingroom128 \
	  test_livingroom192 \

BENCHMARKS=bench_blonde512 \
		   bench_lena512 \
		   bench_livingroom512 \
		   bench_clouds

 
all: $(OBJECTS) $(CUOBJECTS) | $(BINDIR)
  $(CXX) $(OBJECTS) $(CUOBJECTS) -o $(BINDIR)/demo_anlm $(CXXFLAGS) \
      $(CULINK) $(CULDLIBS) $(LDLIBS)

$(OBJDIR)/%.o: %.cpp | $(OBJDIR)
  $(CXX) $< -c -o $@ $(CXXFLAGS) $(MODS)

$(CUOBJDIR)/%.o: %.cu | $(CUOBJDIR)
  $(NVCC) $< -c -o $@ $(CUFLAGS) $(CULDLIBS)

$(OBJDIR):
  mkdir $(OBJDIR)

$(CUOBJDIR):
  mkdir $(CUOBJDIR)

$(BINDIR):
  mkdir $(BINDIR)

clean:
  rm -rf $(OBJDIR) $(CUOBJDIR)

purge: clean
  rm -rf bin

test: $(TESTS)

benchmark: test $(BENCHMARKS)

test_blonde128:
	echo "Running test on blonde128 data..."
	./$(BINDIR)/demo_anlm test_datasets/woman_blonde_small_noisy.karas 6 \
						  test_datasets/woman_blonde_small_filtered.karas

bench_blonde512:
	echo "Running benchmark on blonde512 data..."
	./$(BINDIR)/demo_anlm test_datasets/woman_blonde_noisy.karas 6

test_lena64:
	echo "Running test on lena64 data..."
	./$(BINDIR)/demo_anlm test_datasets/lena_gray_64_noisy.karas 6 \
						  test_datasets/lena_gray_64_filtered.karas

test_lena128:
	echo "Running test on lena128 data..."
	./$(BINDIR)/demo_anlm test_datasets/lena_gray_128_noisy.karas 6 \
						  test_datasets/lena_gray_128_filtered.karas

test_lena192:
	echo "Running test on lena192 data..."
	./$(BINDIR)/demo_anlm test_datasets/lena_gray_192_noisy.karas 6 \
					      test_datasets/lena_gray_192_filtered.karas

bench_lena512:
	echo "Running benchmark on lena512 data..."
	./$(BINDIR)/demo_anlm test_datasets/lena_gray_512_noisy.karas 6

test_livingroom64:
	echo "Running test on livingroom64 data..."
	./bin/demo_anlm test_datasets/livingroom_64_noisy.karas 6 \
					test_datasets/livingroom_64_filtered.karas

test_livingroom128:
	echo "Running test on livingroom128 data..."
	./$(BINDIR)/demo_anlm test_datasets/livingroom_128_noisy.karas 6 \
						  test_datasets/livingroom_128_filtered.karas

test_livingroom192:
	echo "Running test on livingroom192 data..."
	./$(BINDIR)/demo_anlm test_datasets/livingroom_192_noisy.karas 6 \
					      test_datasets/livingroom_192_filtered.karas

bench_livingroom512:
	echo "Running benchmark on livingroom512 data..."
	./$(BINDIR)/demo_anlm test_datasets/livingroom_512_noisy.karas 6

bench_clouds:
	echo "Running benchmark on 'clouds' data..."
	echo "WARNING: This benchmark may take several hours to complete!"
	./$(BINDIR)/demo_anlm test_datasets/clouds_noisy.karas 6
