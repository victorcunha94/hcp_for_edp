#!/bin/bash

# Arquivo de saída
OUTPUT="resultados_weak_scaling.txt"
> $OUTPUT  # limpa arquivo antes de rodar

# Lista de números de processos a testar
PROCS=(1 2 3 4 5 6 7 8 9 10 11 12)

# Loop sobre os números de processos
for p in "${PROCS[@]}"; do
    echo "Rodando com $p processos..." | tee -a $OUTPUT
    mpirun -np $p python3 sum_vector_ws.py weak | tee -a $OUTPUT
    echo "-----------------------------------------------------" | tee -a $OUTPUT
done

