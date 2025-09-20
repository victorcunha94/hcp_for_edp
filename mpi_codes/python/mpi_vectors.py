from mpi4py import MPI
import numpy as np
import time

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    cabeça = 0

    # Tamanho grande para evidenciar paralelismo
    carga = 10 * 1000 * 1000  # 10 milhões de elementos

    # Só o processo cabeça cria os dados originais
    if rank == cabeça:
        print(f"Calculando soma de vetores e produto interno com {carga} elementos")
        print(f"Usando {size} processos MPI")
        
        # Gerar vetores aleatórios
        A = np.random.rand(carga).astype(np.float32)
        B = np.random.rand(carga).astype(np.float32)
        
        # Para verificação dos resultados
        soma_verificacao = A + B
        produto_verificacao = np.dot(A, B)
    else:
        A = B = None
        soma_verificacao = produto_verificacao = None

    # Dividir a carga entre processos (tratando divisão não exata)
    elementos_por_processo = carga // size
    resto = carga % size
    
    # Cada processo descobre quantos elementos vai receber
    if rank < resto:
        N_recebidos = elementos_por_processo + 1
    else:
        N_recebidos = elementos_por_processo

    # Buffers locais para cada processo
    A_local = np.empty(N_recebidos, dtype=np.float32)
    B_local = np.empty(N_recebidos, dtype=np.float32)

    # Medir tempo total de comunicação + computação
    inicio_total = time.time()

    # CORREÇÃO AQUI: usar elementos_por_processo em vez de elements_por_processo
    counts = [elementos_por_processo + 1 if i < resto else elementos_por_processo for i in range(size)]
    displs = [sum(counts[:i]) for i in range(size)]
    
    # Distribuir os dados (Scatterv para divisão não uniforme)
    comm.Scatterv([A, counts, displs, MPI.FLOAT], A_local, root=cabeça)
    comm.Scatterv([B, counts, displs, MPI.FLOAT], B_local, root=cabeça)

    # Medir tempo de cálculo (após comunicação)
    inicio_calculo = time.time()

    # 1. OPERAÇÃO: SOMA DE VETORES (paralela)
    soma_local = A_local + B_local

    # 2. OPERAÇÃO: PRODUTO INTERNO (paralelo + redução)
    produto_local = np.dot(A_local, B_local)

    fim_calculo = time.time()

    # Coletar resultados da soma (opcional, para verificação)
    if rank == cabeça:
        soma_completa = np.empty(carga, dtype=np.float32)
    else:
        soma_completa = None

    comm.Gatherv(soma_local, [soma_completa, counts, displs, MPI.FLOAT], root=cabeça)

    # Reduzir o produto interno para o processo cabeça
    produto_final = comm.reduce(produto_local, op=MPI.SUM, root=cabeça)

    fim_total = time.time()

    # Resultados e análise de performance
    if rank == cabeça:
        print("\n" + "="*60)
        print("RESULTADOS E ANÁLISE DE PERFORMANCE")
        print("="*60)
        
        # Tempos
        tempo_total = fim_total - inicio_total
        tempo_calculo = fim_calculo - inicio_calculo
        tempo_comunicacao = tempo_total - tempo_calculo
        
        print(f"Tempo total: {tempo_total:.4f} segundos")
        print(f"Tempo cálculo: {tempo_calculo:.4f} segundos")
        print(f"Tempo comunicação: {tempo_comunicacao:.4f} segundos")
        print(f"Razão comunicação/cálculo: {tempo_comunicacao/tempo_calculo*100:.1f}%")
        
        # Verificação dos resultados
        erro_soma = np.max(np.abs(soma_completa - soma_verificacao)) if soma_verificacao is not None else 0
        erro_produto = abs(produto_final - produto_verificacao) if produto_verificacao is not None else 0
        
        print(f"\nVerificação:")
        print(f"Erro máximo na soma: {erro_soma:.6e}")
        print(f"Erro no produto interno: {erro_produto:.6e}")
        
        # Métricas de desempenho
        print(f"\nMétricas de desempenho:")
        print(f"Núme    ro de processos: {size}")
        print(f"Elementos totais: {carga}")
        print(f"Elementos por processo: ~{carga/size:.0f}")

if __name__ == "__main__":
    main()
