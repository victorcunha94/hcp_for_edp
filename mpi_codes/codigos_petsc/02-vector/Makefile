include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -pedantic -std=c99
OUTPUT=vector

${OUTPUT}: ${OUTPUT}.o
	  -${CLINKER} -o ${OUTPUT} ${OUTPUT}.o  ${PETSC_LIB}
	   ${RM} ${OUTPUT}.o

run: ${OUTPUT}
	mpirun -n 4 ./$(OUTPUT)

run_vec_view: ${OUTPUT}
	mpirun -n 4 ./$(OUTPUT) -u_vec_view

run_vecbin: ${OUTPUT}
	mpirun -n 4 ./$(OUTPUT) -u_vec_view binary:vec.dat

run_profiling: ${OUTPUT}
	mpirun -n 4 ./$(OUTPUT) -log_view
		
clean::
	@rm -f *~ $(OUTPUT) *.o *tmp
	@rm -f *.dat *.dat.info
