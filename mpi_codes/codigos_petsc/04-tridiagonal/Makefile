include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -pedantic -std=c99
OUTPUT=tri

${OUTPUT}: ${OUTPUT}.o
	  -${CLINKER} -o ${OUTPUT} ${OUTPUT}.o  ${PETSC_LIB}
	   ${RM} ${OUTPUT}.o

run_no_pc: ${OUTPUT}
	mpirun -n 4 ./$(OUTPUT) -m 10000  -ksp_type cg -pc_type none -ksp_rtol 1e-12 -ksp_max_it 10000  -ksp_converged_reason -ksp_monitor ascii:res.txt -log_view

run_sor: ${OUTPUT}
	mpirun -n 4 ./$(OUTPUT) -m 10000 -ksp_type cg -pc_type sor -pc_sor_omega 1.0 -ksp_rtol 1e-12 -ksp_max_it 10000 -ksp_converged_reason -ksp_monitor ascii:res.txt -ksp_view

run_jacobi: ${OUTPUT}
	mpirun -n 4 ./$(OUTPUT) -m 10000  -ksp_type cg -pc_type jacobi -ksp_rtol 1e-12 -ksp_max_it 10000 -ksp_converged_reason -ksp_monitor ascii:res.txt -ksp_view

run_bjacobi: ${OUTPUT}
	mpirun -n 4 ./$(OUTPUT) -m 10000  -ksp_type cg -pc_type bjacobi -ksp_rtol 1e-12 -ksp_max_it 10000 -ksp_converged_reason -ksp_monitor ascii:res.txt -ksp_view

clean::
	@rm -f *~ $(OUTPUT) *.o *tmp
	@rm -f *.dat *.dat.info
