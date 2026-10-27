# set _OPENMP to enable openmp support, currently not
include ../Makefile.inc 

CFILES = $(shell ls | grep .c)
OBJC = $(CFILES:.c=.o)

.PHONY : all clean

all: $(OBJC)
	ar rcs libsymspg.a $(OBJC)
	mv libsymspg.a ..
clean: 
	-rm $(OBJC)

