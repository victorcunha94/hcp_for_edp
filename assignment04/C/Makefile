PETSC_DIR = /usr/lib/petsc
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

CFLAGS += -std=c99
OUTPUT=Main

debug: CFLAGS += -g -O0 -DDEBUG_MODE -Wall -Wextra
debug: debug_$(OUTPUT)
debug_$(OUTPUT).o: $(OUTPUT).c
	${CC} -c -o $@ $^ ${CFLAGS} ${PETSC_CC_INCLUDES}
debug_sol.o: Sol_exata.c
	${CC} -c -o $@ $^ ${CFLAGS} ${PETSC_CC_INCLUDES}
debug_$(OUTPUT): debug_$(OUTPUT).o debug_sol.o
	${CLINKER} -o $@ $^ ${PETSC_LIB} -lm

release: CFLAGS += -O3 -DNDEBUG -Wall
release: release_$(OUTPUT)
release_$(OUTPUT).o: $(OUTPUT).c
	${CC} -c -o $@ $^ ${CFLAGS} ${PETSC_CC_INCLUDES}
release_sol.o: Sol_exata.c
	${CC} -c -o $@ $^ ${CFLAGS} ${PETSC_CC_INCLUDES}
release_$(OUTPUT): release_$(OUTPUT).o release_sol.o
	${CLINKER} -o $@ $^ ${PETSC_LIB} -lm

run: debug_${OUTPUT}
	mpirun --mca btl_tcp_if_include lo ./debug_Main -log_view -log_view_memory -malloc_dump

clean:
	@rm -f *~ $(OUTPUT) debug_$(OUTPUT) release_$(OUTPUT) *.o *tmp
	@rm -f *.dat *.dat.info
