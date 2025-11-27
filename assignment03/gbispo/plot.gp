set title "Solução Numérica da Equação de Poisson"
set xlabel "Eixo X"
set ylabel "Eixo Y"

# Configurações para plotagem 3D
set dgrid3d 50,50 # Interpola os dados em uma grade 50x50
set pm3d at s    # Desenha uma superficie colorida
set hidden3d    # Esconde linhas de grade na parte de tras

# Rotaciona o grafico para uma visualizacao melhor
set view 60, 30, 1.2

# Cria uma paleta de cores para o grafico
set palette defined (-1 "blue", 0 "white", 1 "red")

# Plota a superficie a partir do arquivo
splot "solucao_serial.dat" with pm3d

# Salva o grafico em um arquivo PNG
set terminal pngcairo
set output "solucao_serial.png"
# Ajusta as margens para dar mais espaco
set lmargin 8
set rmargin 4
set tmargin 2
set bmargin 4

# Ajusta a posicao do titulo
set title "Solução Numérica da Equação de Poisson" offset 0, -1

# Reposiciona a legenda para o canto superior esquerdo
set key outside right top

# Rotaciona os eixos para uma visualizacao melhor
set view 60, 30, 1.2

# Estilo de plotagem
set pm3d at s
set hidden3d
set palette defined (-1 "blue", 0 "white", 1 "red")

set xlabel "Eixo X" offset 0,0, 0
set ylabel "Eixo Y" offset 0,0, 0
set zlabel "Valor de u" offset 0,0, 0

# Plota a superficie a partir do arquivo
splot "solucao_serial.dat" with pm3d
replot

# Limpa as configuracoes para nao interferir em outros plots
unset terminal