CXX = g++
CXXFLAGS = -march=native -fPIC -I/home/atao/workspace/codes/atoms/deplibs/fmt/include -O3 -DNDEBUG -fvisibility=hidden -fvisibility-inlines-hidden

CPPFILES := $(shell ls | grep .cc)
OBJS = $(CPPFILES:.cc=.o)

.PHONY : all clean

all: $(OBJS)
	ar rs libfmt.a $(OBJS)
	mv libfmt.a ../
clean: 
	-rm $(OBJS) 

