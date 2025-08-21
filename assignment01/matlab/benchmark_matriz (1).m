function benchmark_matriz()
    sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000];
    
    % Inicializar vetores para tempos
    time_ijk = zeros(size(sizes));
    time_ikj = zeros(size(sizes));
    time_built_in = zeros(size(sizes));
    
    for idx = 1:length(sizes)
        N = sizes(idx);
        fprintf('\n--- Matriz %dx%d ---\n', N, N);
        
        A = rand(N,N);
        B = rand(N,N);
        C = zeros(N,N);
        
        %% Ordem ijk
        tic;
        for i = 1:N
            for j = 1:N
                soma = 0;
                for k = 1:N
                    soma = soma + A(i,k) * B(k,j); % Correção: A(i,k) em vez de A(i,j)
                end
                C(i,j) = soma;
            end
        end
        time_ijk(idx) = toc;
        fprintf('Ordem i-j-k: %.4f s\n', time_ijk(idx));
        
        %% Ordem i-k-j
        C = zeros(N,N);
        tic;
        for i = 1:N
            for k = 1:N
                aik = A(i,k);
                for j = 1:N
                    C(i,j) = C(i,j) + aik * B(k,j);
                end
            end
        end
        time_ikj(idx) = toc;
        fprintf('Ordem i-k-j: %.4f s\n', time_ikj(idx));
        
        %% Função nativa MATLAB (BLAS)
        tic;
        C = A * B; % Multiplicação nativa otimizada
        time_built_in(idx) = toc;
        fprintf('Multiplicação nativa MATLAB (BLAS): %.4f s\n', time_built_in(idx));
    end
    
    %% Plotar os resultados
    figure;
    plot(sizes, time_ijk, '-o', 'LineWidth', 2); hold on;
    plot(sizes, time_ikj, '-s', 'LineWidth', 2);
    plot(sizes, time_built_in, '-d', 'LineWidth', 2);
    grid on;
    xlabel('Tamanho da matriz N');
    ylabel('Tempo (s)');
    title('Comparação de tempos de multiplicação de matrizes em MATLAB');
    legend('i-j-k', 'i-k-j', 'Função nativa (BLAS)', 'Location','northwest');
    
end
