include ../Makefile.inc 

CPPFILES := $(shell ls | grep .cpp)
OBJS = $(CPPFILES:.cpp=.o)

.PHONY : all clean
all: $(OBJS)
	ar rs libtinyxml2.a $^
	mv libtinyxml2.a ..
clean: 
	-rm $(OBJS) $(OBJS:.o) 

