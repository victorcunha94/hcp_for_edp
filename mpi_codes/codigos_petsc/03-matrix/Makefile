include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -pedantic -std=c99
OUTPUT=matrix

${OUTPUT}: ${OUTPUT}.o
	      -${CLINKER} -o ${OUTPUT} ${OUTPUT}.o  ${PETSC_LIB}
	       ${RM} ${OUTPUT}.o

run: ${OUTPUT}
	mpirun -n 4 ./$(OUTPUT) -vec_view -mat_view ::ascii_dense  -m 5

run_draw: ${OUTPUT}
	mpirun -n 4 ./$(OUTPUT) -mat_view draw -draw_pause 5 -m 100

run_savebin: ${OUTPUT}
	mpirun -n 4 ./$(OUTPUT) -mat_view binary:mat.dat

run_savebin_matlab: ${OUTPUT}
	mpirun -n 4 ./$(OUTPUT) -mat_view ascii:mat.m:ascii_matlab

clean::
	@rm -f *~ $(OUTPUT) *.o *tmp
	@rm -f *.dat *.dat.info
