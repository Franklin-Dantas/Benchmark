import psutil
import numpy as np
import time
 
# Tentar importar cupy para uso de GPU, se disponível
try:
    import cupy as cp
    gpu_available = True
except ImportError:
    gpu_available = False
 
# Função para realizar o benchmark de multiplicação de matrizes e medir o desempenho
def benchmark_matrix_multiplication_with_metrics(size, use_gpu=False):
    # Pega o uso de memória e CPU iniciais
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024 ** 2)  # Convertendo para MB
    initial_cpu = psutil.cpu_percent(interval=None)
 
    # Gera duas matrizes aleatórias grandes (na CPU ou GPU)
    if use_gpu and gpu_available:
        matrix_a = cp.random.rand(size, size)
        matrix_b = cp.random.rand(size, size)
    else:
        matrix_a = np.random.rand(size, size)
        matrix_b = np.random.rand(size, size)
 
    # Começa a contagem de tempo
    start_time = time.time()
 
    # Realiza a multiplicação de matrizes (na CPU ou GPU)
    if use_gpu and gpu_available:
        result = cp.dot(matrix_a, matrix_b)
        cp.cuda.Stream.null.synchronize()  # Sincroniza para garantir que a operação GPU terminou
    else:
        result = np.dot(matrix_a, matrix_b)
 
    # Termina a contagem de tempo
    end_time = time.time()
 
    # Pega o uso de memória e CPU finais
    final_memory = process.memory_info().rss / (1024 ** 2)  # Convertendo para MB
    final_cpu = psutil.cpu_percent(interval=None)
 
    # Calcula o uso de memória e tempo gasto
    memory_consumed = final_memory - initial_memory
    time_taken = end_time - start_time
 
    # Calcula o número de operações (aproximado)
    operations = size ** 3  # Número de multiplicações realizadas
    operations_per_second = operations / time_taken
 
    # Retorna as métricas relevantes do benchmark
    return {
        "matrix_size": size,
        "time_taken": time_taken,
        "memory_consumed_mb": memory_consumed,
        "cpu_usage_percent": final_cpu,
        "operations_per_second": operations_per_second,
        "gpu_used": use_gpu and gpu_available
    }
 
# Tamanhos de matrizes para estressar o sistema
matrix_sizes = [1000, 2000, 3000, 4000]  # Tamanhos grandes para CPU/GPU
 
# Armazenar resultados
benchmark_results = []
 
# Rodar o benchmark tanto na CPU quanto na GPU (se disponível)
for size in matrix_sizes:
    # CPU Benchmark
    result_cpu = benchmark_matrix_multiplication_with_metrics(size, use_gpu=False)
    benchmark_results.append(result_cpu)
 
    # GPU Benchmark, se GPU estiver disponível
    if gpu_available:
        result_gpu = benchmark_matrix_multiplication_with_metrics(size, use_gpu=True)
        benchmark_results.append(result_gpu)
 
# Exibir os resultados
import pandas as pd
df_results = pd.DataFrame(benchmark_results)
print(df_results)
