import numpy as np
from mpi4py import MPI

def dot_product(A, B):
    """Calcula produto interno entre dois vetores"""
    result = 0.0
    for i in range(len(A)):
        result += A[i] * B[i]
    return result

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        # Vetores de entrada
        A = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
        B = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=np.float64)
        
        print(f"Vetor A: {A}")
        print(f"Vetor B: {B}")
        
        if size == 1:
            result = dot_product(A, B)
            print(f"Produto interno: {result}")
            return
        
        # Dividir trabalho
        num_workers = size - 1
        chunk_size = len(A) // num_workers
        remainder = len(A) % num_workers

        start = 0
        requests_send = []  # Lista para armazenar requests de envio
        
        for i in range(1, size):
            end = start + chunk_size
            if (i - 1) < remainder:
                end += 1

            A_chunk = A[start:end]
            B_chunk = B[start:end]

            # Envio não-bloqueante
            req_send = comm.isend(('dot_product', A_chunk, B_chunk), dest=i, tag=1)
            requests_send.append(req_send)
            
            print(f"Mestre enviou para worker {i}: {len(A_chunk)} elementos")
            start = end

        # Esperar todos os envios completarem
        MPI.Request.Waitall(requests_send)
        print("Todos os envios foram completados")

        # Recepção não-bloqueante
        requests_recv = []
        for i in range(1, size):
            req_recv = comm.irecv(source=i, tag=2)
            requests_recv.append(req_recv)

        # Coletar resultados
        total_result = 0.0
        for i, req in enumerate(requests_recv, 1):
            partial_result = req.wait()  # Esperar por cada resultado
            total_result += partial_result
            print(f"Mestre recebeu do worker {i}: {partial_result}")
            
        print(f"Produto interno final: {total_result}") 

    else:
        # Workers
        print(f"Worker {rank}: Aguardando dados...")
        
        # Receber operação e dados (bloqueante para workers)
        operation, A_chunk, B_chunk = comm.recv(source=0, tag=1)
        
        if operation == 'dot_product':
            result = dot_product(A_chunk, B_chunk)
            print(f"Worker {rank}: calculou produto parcial = {result}")
            
            # Envio não-bloqueante de volta para mestre
            req_send = comm.isend(result, dest=0, tag=2)
            req_send.wait()  # Esperar envio completar
            print(f"Worker {rank}: resultado enviado")
        else:
            print(f"Worker {rank}: Operação desconhecida")

if __name__ == "__main__":
    main()
