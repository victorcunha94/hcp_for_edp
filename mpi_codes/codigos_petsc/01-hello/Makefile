include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -pedantic -std=c99
OUTPUT=hello

${OUTPUT}: ${OUTPUT}.o
	  -${CLINKER} -o ${OUTPUT} ${OUTPUT}.o  ${PETSC_LIB}
	   ${RM} ${OUTPUT}.o

run: ${OUTPUT}
	mpirun -n 4 ./$(OUTPUT)

clean::
	@rm -f *~ $(OUTPUT) *.o *tmp
	@rm -f *.dat *.dat.info
