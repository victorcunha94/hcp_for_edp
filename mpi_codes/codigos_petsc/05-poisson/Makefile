include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -pedantic -std=c99
OUTPUT=poisson

${OUTPUT}: ${OUTPUT}.o
	  -${CLINKER} -o ${OUTPUT} ${OUTPUT}.o  ${PETSC_LIB}
	   ${RM} ${OUTPUT}.o

run_cg_no_pc: ${OUTPUT}
	mpirun -n 4 ./$(OUTPUT) -da_grid_x 169 -da_grid_y 160 -ksp_type cg -pc_type none -ksp_rtol 1e-12 -ksp_max_it 10000  -ksp_converged_reason -ksp_view

run_cg_pc: ${OUTPUT}
	mpirun -n 4 ./$(OUTPUT) -da_grid_x 160 -da_grid_y 160 -ksp_type cg -pc_type bjacobi -ksp_rtol 1e-12 -ksp_max_it 10000  -ksp_converged_reason -ksp_view


clean::
	@rm -f *~ $(OUTPUT) *.o *tmp
	@rm -f *.dat *.dat.info
