function benchmark_matriz()
    sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000];
    
    % Inicializar vetores para tempos
    time_jik = zeros(size(sizes));
    time_ikj = zeros(size(sizes));
    time_jki = zeros(size(sizes));
    time_built_in = zeros(size(sizes));
    
    for idx = 1:length(sizes)
        N = sizes(idx);
        fprintf('\n--- Matriz %dx%d ---\n', N, N);
        
        A = rand(N,N);
        B = rand(N,N);
        C = zeros(N,N);
        
        %% Ordem j-i-k
        tic;
        for j = 1:N
            for i = 1:N
                soma = 0;
                for k = 1:N
                    soma = soma + A(i,k) * B(k,j);
                end
                C(i,j) = soma;
            end
        end
        time_jik(idx) = toc;
        fprintf('Ordem j-i-k: %.4f s\n', time_jik(idx));
        
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
        
        %% Ordem j-k-i
        C = zeros(N,N);
        tic;
        for j = 1:N
            for k = 1:N
                bkj = B(k,j);
                for i = 1:N
                    C(i,j) = C(i,j) + A(i,k) * bkj;
                end
            end
        end
        time_jki(idx) = toc;
        fprintf('Ordem j-k-i: %.4f s\n', time_jki(idx));
        
        %% Função nativa MATLAB (BLAS)
        C = zeros(N,N);
        tic;
        C = A * B;
        time_built_in(idx) = toc;
        fprintf('Multiplicação nativa MATLAB: %.4f s\n', time_built_in(idx));
    end
    
    %% Plotar os resultados
    figure;
    plot(sizes, time_jik, '-o', 'LineWidth', 2); hold on;
    plot(sizes, time_ikj, '-s', 'LineWidth', 2);
    plot(sizes, time_jki, '-^', 'LineWidth', 2);
    plot(sizes, time_built_in, '-d', 'LineWidth', 2);
    grid on;
    xlabel('Tamanho da matriz N');
    ylabel('Tempo (s)');
    title('Comparação de tempos de multiplicação de matrizes em MATLAB');
    legend('j-i-k', 'i-k-j', 'j-k-i', 'Função nativa (BLAS)', 'Location','northwest');
end
