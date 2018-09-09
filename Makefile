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
				DMat.o)

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
