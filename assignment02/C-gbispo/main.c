#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <omp.h>
/*
// Euler: analitico (R(z) = 1 + z)
static inline double complex amplification(double complex z) {
    return 1.0 + z;
}

static inline int is_stable_analytic(double complex z) {
    return cabs(amplification(z)) <= 1.0;
}
*/

//Euler: numerico (forca bruta)
static inline int is_stable_euler(double complex u, double complex z, double tolsup, double tolinf){
    for (int i=0; i < 1000; i++){
        u = (1 + z)*u;
        double mag = cabs(u);
        if (mag>tolsup){
            return 0;
        }
        if (mag<tolinf){
            return 1;
        }
    }
    return 0;
}




// Save stability region as PPM image
void save_ppm(const char *filename, int Nx, int Ny, int *stability,
              double x_min, double x_max, double y_min, double y_max) {
    FILE *f = fopen(filename, "w");
    if (!f) { perror("fopen"); return; }

    fprintf(f, "P3\n%d %d\n255\n", Nx, Ny);

    double dx = (x_max - x_min) / (Nx - 1);
    double dy = (y_max - y_min) / (Ny - 1);
    double grid_step = 1.0;  // grid spacing

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
            } else if (val) {
                fprintf(f, "0 0 0 ");    // stable = black
            } else {
                fprintf(f, "255 255 255 "); // unstable = white
            }
        }
        fprintf(f, "\n");
    }

    fclose(f);
}

int main() {
    double tolsup = 1e6;
    double tolinf = 1e-6;
    double h = 1e-3;
    double x_min = -5.0, x_max = 5.0;
    double y_min = -5.0, y_max = 5.0;
    int Nx = (int) ((x_max - x_min)/h) + 1, Ny =(int) ((y_max - y_min)/h) + 1;   // Resolucao

    int *stability = malloc(Nx * Ny * sizeof(int));
    if (!stability) {
        perror("malloc failed");
        return 1;
    }

    // Paralelizacao euler analitico
    /*
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < Nx; i++) {
       double x = x_min + i * (x_max - x_min) / (Nx - 1);
       for (int j = 0; j < Ny; j++) {
            double y = y_min + j * (y_max - y_min) / (Ny - 1);
            double complex z = x + I*y;
            stability[i * Ny + j] = is_stable(z);
      }
    }
    */

    //Forca bruta Euler
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < Nx; i++) {
       double x = x_min + i * h;
       for (int j = 0; j < Ny; j++) {
            double y = y_min + j * h;
            double complex z = x + I*y;
            stability[i * Ny + j] = is_stable_euler(1.0, z, tolsup, tolinf);
       }
    }
    save_ppm("stability.ppm", Nx, Ny, stability, x_min, x_max, y_min, y_max);

    free(stability);
    return 0;
}
