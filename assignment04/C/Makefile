PETSC_DIR = /usr/lib/petsc
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -std=c99
OUTPUT=Main

debug: CFLAGS += -g -O0 -DDEBUG_MODE -Wall -Wextra
debug: debug_$(OUTPUT)
debug_$(OUTPUT).o: $(OUTPUT).c
	-${CC} -c -o $@ $(OUTPUT).c ${CFLAGS} ${PETSC_CC_INCLUDES}
debug_$(OUTPUT): debug_$(OUTPUT).o
	-${CLINKER} -o $@ $^ ${PETSC_LIB}
	${RM} $^

release: CFLAGS += RELEASE_CFLAGS = -O3 -DNDEBUG -Wall
release: release_$(OUTPUT)
release_$(OUTPUT).o: $(OUTPUT).c
	-${CC} -c -o $@ $(OUTPUT).c ${CFLAGS} ${PETSC_CC_INCLUDES}
release_$(OUTPUT): release_$(OUTPUT).o
	-${CLINKER} -o $@ $^ ${PETSC_LIB}
	${RM} $^

run: debug_${OUTPUT}
	mpirun --mca btl_tcp_if_include lo ./debug_Main -log_view -log_view_memory -malloc_dump
#::ascii_flamegraph -log_view_memory | flamegraph | display caso queira fazer flamegraph bonito

clean:
	@rm -f *~ $(OUTPUT) debug_$(OUTPUT) release_$(OUTPUT) *.o *tmp
	@rm -f *.dat *.dat.info