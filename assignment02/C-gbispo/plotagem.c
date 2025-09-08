//
// Created by bruno on 07/09/25.
//
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <omp.h>
#include "metodos_numericos.h"
#include "plotagem.h"
void make_filename(char *dest, size_t size,
                   const char *basename, const char *ext) {
    snprintf(dest, size, "%s%s", basename, ext);
}


void save_ppm(const char *filename, int Nx, int Ny, int *stability,
              double x_min, double x_max, double y_min, double y_max) {
    FILE *f = fopen(filename, "w");
    if (!f) { perror("fopen"); return; }

    fprintf(f, "P3\n%d %d\n255\n", Nx, Ny);

    double dx = (x_max - x_min) / (Nx - 1);
    double dy = (y_max - y_min) / (Ny - 1);
    double grid_step = 1.0;  // grid spacing for red lines

    for (int j = Ny - 1; j >= 0; j--) {
        double y = y_min + j * dy;
        for (int i = 0; i < Nx; i++) {
            double x = x_min + i * dx;

            int val = stability[i * Ny + j];
            // nearest grid points
            double gx = round(x / grid_step) * grid_step;
            double gy = round(y / grid_step) * grid_step;

            int on_vertical   = fabs(x - gx) < dx/2;
            int on_horizontal = fabs(y - gy) < dy/2;

            int on_y_axis = fabs(x) < dx/2;
            int on_x_axis = fabs(y) < dy/2;

            if (on_x_axis || on_y_axis) {
                fprintf(f, "0 0 255 ");   // axes = blue
            } else if (on_vertical || on_horizontal) {
                fprintf(f, "255 0 0 ");  // grid = red

            } else {
                switch (val)
                {
                case 0:
                    fprintf(f, "0 0 0 ");    // stable = black
                    break;

                case 1:
                    fprintf(f, "0 255 0 ");    // stable = black
                    break;

                case 2:
                    fprintf(f, "0 0 255 ");    // stable = black
                    break;
                case 3:
                    fprintf(f, "255 0 0 ");    // stable = black
                    break;
                case 4:
                    fprintf(f, "120 120 0 ");    // stable = black
                    break;
                case 5:
                    fprintf(f, "120 0 120 ");    // stable = black
                    break;
                case 6:
                    fprintf(f, "0 120 120 ");    // stable = black
                    break;
                case 7:
                    fprintf(f, "120 120 120 ");    // stable = black
                    break;
                case -1:
                    fprintf(f, "255 255 255 ");    // stable = black
                    break;
                default:
                    break;
                }
            }
        }
        fprintf(f, "\n");
    }

    fclose(f);
}

void save_csv(const char *filename, int Nx, int Ny, int *stability,
              double x_min, double x_max, double y_min, double y_max) {
    FILE *f = fopen(filename, "w");
    if (!f) { perror("fopen"); return; }

    // header
    fprintf(f, "x,y,stable\n");

    double dx = (x_max - x_min) / (Nx - 1);
    double dy = (y_max - y_min) / (Ny - 1);

    for (int i = 0; i < Nx; i++) {
        double x = x_min + i * dx;
        for (int j = 0; j < Ny; j++) {
            double y = y_min + j * dy;
            int val = stability[i * Ny + j];
            fprintf(f, "%f,%f,%d\n", x, y, val);
        }
    }

    fclose(f);
}

void plot(const char *filename, int Nx, int Ny,
                      double x_min, double y_min, double h,
                      int method, int n_threads) {
    int *stability = malloc(Nx * Ny * sizeof(int));
    if (!stability) { perror("malloc failed"); exit(1); }

    double tolsup = 1e6;
    double tolinf = 1e-6;

    #pragma omp parallel for schedule(dynamic) num_threads(n_threads)
    for (int j = 0; j < Ny; j++) {
        double y = y_min + j * h;
        for (int i = 0; i < Nx; i++) {
            int tid = omp_get_thread_num();
            double x = x_min + i * h;
            double complex z = x + I*y;

            switch (method) {
                case 0: stability[i*Ny+j] = is_stable(euler_amp(z)); break;
                case 1: stability[i*Ny+j] = is_stable_euler_bruteforce(z,tolsup,tolinf); break;
                case 2: stability[i*Ny+j] = is_stable(rk2_amp(z)); break;
                case 3: stability[i*Ny+j] = is_stable(rk3_amp(z)); break;
                case 4: stability[i*Ny+j] = is_stable(rk4_amp(z)); break;
                case 5: stability[i*Ny+j] = is_stable_rk4_bruteforce(z,tolsup,tolinf,tid); break;
                case 6: stability[i*Ny+j] = is_stable_rk3_bruteforce(z,tolsup,tolinf); break;
                case 7: stability[i*Ny+j] = is_stable_eulerimplicit_bruteforce(z,tolsup,tolinf,tid); break;
                case 8: stability[i*Ny+j] = is_stable_trapez(z,tolsup,tolinf,tid); break;
                case 9: stability[i*Ny+j] = is_stable_ab2(z,tolsup,tolinf,tid); break;
                case 10: stability[i*Ny+j] = is_stable_ab3(z,tolsup,tolinf,tid); break;
                case 11: stability[i*Ny+j] = is_stable_ab4(z,tolsup,tolinf,tid); break;
                case 12: stability[i*Ny+j] = is_stable_ab5(z,tolsup,tolinf,tid); break;
                case 13: stability[i*Ny+j] = is_stable_am2(z,tolsup,tolinf,tid); break;
                case 14: stability[i*Ny+j] = is_stable_am3(z,tolsup,tolinf,tid); break;
                case 15: stability[i*Ny+j] = is_stable_am4(z,tolsup,tolinf,tid); break;
                case 16: stability[i*Ny+j] = is_stable_am5(z,tolsup,tolinf,tid); break;

                default: stability[i*Ny+j] = 0;
            }
        }
    }
    char csvfile[128], ppmfile[128];
    make_filename(csvfile, sizeof(csvfile), filename, ".csv");
    make_filename(ppmfile, sizeof(ppmfile), filename, ".ppm");

    save_ppm(ppmfile, Nx, Ny, stability,
             x_min, x_min + (Nx-1)*h,
             y_min, y_min + (Ny-1)*h);

    save_csv(csvfile, Nx, Ny, stability,
             x_min, x_min + (Nx-1)*h,
             y_min, y_min + (Ny-1)*h);

    free(stability);
}