#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <omp.h>
#include <time.h>

// =======================================================
// Fatores de amplificacao: R(z)
// =======================================================

static inline int is_stable(double complex z) {

    return cabs(z) <= 1.0;
}
// Euler 
static inline double complex euler_amp(double complex z) {
    return 1.0 + z;
}


// RK2 
static inline double complex rk2_amp(double complex z) {
    return 1.0 + z + (cpow(z,2))/2.0;
}

// RK3 
static inline double complex rk3_amp(double complex z) {
    return 1.0 + z + (cpow(z,2))/2.0 + (cpow(z,3))/6.0;
}


// RK4 
static inline double complex rk4_amp(double complex z) {
    return 1.0 + z + (cpow(z,2))/2.0 + (cpow(z,3))/6.0 + (cpow(z,4))/24.0;
}
// =======================================================
// Forca Bruta
// =======================================================

// Euler forca bruta
static inline int is_stable_euler_bruteforce(double complex z,
                                             double tolsup, double tolinf) {
    double complex u = 1.0;
    for (int i = 0; i < 1000; i++) {
        u = (1.0 + z) * u;
        double mag = cabs(u);
        if (mag > tolsup) return 0; // instavel
        if (mag < tolinf) return 1; // estavel
    }
    return 0; 
}

static inline int is_stable_rk3_bruteforce(double complex z,
                                           double tolsup, double tolinf) {
    double complex u = 1.0;
    for (int i = 0; i < 3000; i++) {
        // RK4 step
        double complex k1 = z * u;
        double complex k2 = z * (u + k1/3.0);
        double complex k3 = z * (u + 2*k2/3.0);
        u = u + (k1 + 3.0*k3)/4.0;

        double mag = cabs(u);
        if (mag > tolsup) return 0; 
        if (mag < tolinf) return 1; 
    }
    return 0; 
}

static inline int is_stable_rk4_bruteforce(double complex z,
                                           double tolsup, double tolinf, int tid) {
    double complex u = 1.0;
    for (int i = 0; i < 3000; i++) {
        // RK4 step
        double complex k1 = z * u;
        double complex k2 = z * (u + 0.5*k1);
        double complex k3 = z * (u + 0.5*k2);
        double complex k4 = z * (u + k3);
        u = u + (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0;

        double mag = cabs(u);
        if (mag > tolsup) return 8; 
        if (mag < tolinf) return tid; 
    }
    return 8; 
}


// =======================================================
// Salvar um .ppm
// =======================================================
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
            } /*else if (on_vertical || on_horizontal) {
                fprintf(f, "255 0 0 ");  // grid = red

            }*/ else {
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
                case 8:
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

// =======================================================
// Plotar
// =======================================================
void plot(const char *filename, int Nx, int Ny,
                      double x_min, double y_min, double h,
                      int method) {
    int *stability = malloc(Nx * Ny * sizeof(int));
    if (!stability) { perror("malloc failed"); exit(1); }

    double tolsup = 1e6;  
    double tolinf = 1e-6;

    #pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < Ny; j++) {
        double y = y_min + j * h;
    //printf("%d \n", omp_get_thread_num());
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
                default: stability[i*Ny+j] = 0;
            }
        }
    }

    save_ppm(filename, Nx, Ny, stability,
             x_min, x_min + (Nx-1)*h,
             y_min, y_min + (Ny-1)*h);

    free(stability);
}

// =======================================================
// Main 
// =======================================================
int main() {
    double h = 0.005;
    double x_min = -5.0, x_max = 5.0;
    double y_min = -5.0, y_max = 5.0;
    int Nx = (int)((x_max - x_min)/h) + 1;
    int Ny = (int)((y_max - y_min)/h) + 1;
    long int ini = clock();
    //plot("stability_euler.ppm", Nx, Ny, x_min, y_min, h, 0);
    //plot("stability_euler_bruteforce.ppm",Nx, Ny, x_min, y_min, h, 1);
    //plot("stability_rk2.ppm", Nx, Ny, x_min, y_min, h, 2);
    //plot("stability_rk3.ppm", Nx, Ny, x_min, y_min, h, 3);
    //plot("stability_rk4.ppm", Nx, Ny, x_min, y_min, h, 4);
    plot("stability_rk4_bruteforce.ppm",Nx, Ny, x_min, y_min, h, 5 );
    //plot("stability_rk3_bruteforce.ppm",Nx, Ny, x_min, y_min, h, 6 );
    long int end = clock();

    printf("%ld", end-ini);
    return 0;
}
