//
// Created by bruno on 07/09/25.
//

#ifndef PLOTAGEM_H
#define PLOTAGEM_H
#include <stdio.h>
//Gerar os nomes com extensao
void make_filename(char *dest, size_t size,
                   const char *basename, const char *ext);
// Salvar um .ppm
// =======================================================
void save_ppm(const char *filename, int Nx, int Ny, int *stability,
              double x_min, double x_max, double y_min, double y_max);
// =======================================================
// Salvar um .csv
// =======================================================
void save_csv(const char *filename, int Nx, int Ny, int *stability,
              double x_min, double x_max, double y_min, double y_max);
// =======================================================
// Plotar
// =======================================================
void plot(const char *filename, int Nx, int Ny,
                      double x_min, double y_min, double h,
                      int method);
#endif //PLOTAGEM_H
