#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <string.h>

//Gerar os nomes com extensao
void make_filename(char *dest, size_t size,
                   const char *basename, const char *ext) {
    snprintf(dest, size, "%s%s", basename, ext);
}

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
static inline int is_stable_eulerimplicit_bruteforce(double complex z,
                                             double tolsup, double tolinf, int tid) {
    double complex u = 1.0;
    for (int i = 0; i < 1000; i++) {
        u = u/(1.0 - z);
        double mag = cabs(u);
        if (mag > tolsup) return -1; // instavel
        if (mag < tolinf) return tid; // estavel
    }
    return -1;
}

static inline int is_stable_rk3_bruteforce(double complex z,
                                           double tolsup, double tolinf) {
    double complex u = 1.0;
    for (int i = 0; i < 3000; i++) {
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
        double complex k1 = z * u;
        double complex k2 = z * (u + 0.5*k1);
        double complex k3 = z * (u + 0.5*k2);
        double complex k4 = z * (u + k3);
        u = u + (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0;

        double mag = cabs(u);
        if (mag > tolsup) return -1;
        if (mag < tolinf) return tid;
    }
    return -1;
}

static inline int is_stable_trapez(double complex z,
                                           double tolsup, double tolinf, int tid) {
    double complex u = 1.0;
    for (int i = 0; i < 3000; i++) {
        double complex k1 = 1 + z/2;
        double complex k2 = 1 - z/2;
        u = u*(k1/k2);

        double mag = cabs(u);
        if (mag > tolsup) return -1;
        if (mag < tolinf) return tid; 
    }
    return -1;
}

static inline int is_stable_ab2(double complex z,
                                           double tolsup, double tolinf, int tid) {
    double complex u = 1.0;
    double complex unp1 = (1+z)*u;
    for (int i = 0; i < 3000; i++) {
        double complex unp2 = unp1 + (z/2)*(3.0*unp1 - u);
        u = unp1;
        unp1 = unp2;

        double mag = cabs(unp2);
        if (mag > tolsup) return -1;
        if (mag < tolinf) return tid;
    }
    return -1;
}

static inline int is_stable_ab3(double complex z,
                                           double tolsup, double tolinf, int tid) {
    double complex u = 1.0;
    double complex unp1 = (1+z)*u;
    double complex unp2 = unp1 + (z/2)*(3.0*unp1 - u);
    for (int i = 0; i < 10000; i++) {
        double complex unp3 = unp2 + (z/12)*(23.0*unp2 - 16.0*unp1 + 5.0*u);
        u = unp1;
        unp1 = unp2;
        unp2 = unp3;
        double mag = cabs(unp3);
        if (mag > tolsup) return -1;
        if (mag < tolinf) return tid;
    }
    return -1;
}
static inline int is_stable_ab4(double complex z,
                                           double tolsup, double tolinf, int tid) {
    double complex u = 1.0;
    double complex unp1 = (1+z)*u;
    double complex unp2 = unp1 + (z/2)*(3.0*unp1 - u);
    double complex unp3 = unp2 + (z/12)*(23.0*unp2 - 16.0*unp1 + 5.0*u);
    for (int i = 0; i < 10000; i++) {
        double complex unp4 = unp3 + (z/24)*(55.0*unp3 - 59.0*unp2 + 37.0*unp1 - 9.0*u);
        u = unp1;
        unp1 = unp2;
        unp2 = unp3;
        unp3 = unp4;
        double mag = cabs(unp4);
        if (mag > tolsup) return -1;
        if (mag < tolinf) return tid;
    }
    return -1;
}
static inline int is_stable_ab5(double complex z,
                                           double tolsup, double tolinf, int tid) {
    double complex u = 1.0;
    double complex unp1 = (1+z)*u;
    double complex unp2 = unp1 + (z/2)*(3.0*unp1 - u);
    double complex unp3 = unp2 + (z/12)*(23.0*unp2 - 16.0*unp1 + 5.0*u);
    double complex unp4 = unp3 + (z/24)*(55.0*unp3 - 59.0*unp2 + 37.0*unp1 - 9.0*u);
    for (int i = 0; i < 10000; i++) {
        double complex unp5 = unp4 + (z/720)*(1901.0*unp4 - 2774.0*unp3 + 2616.0*unp2 - 1274.0*unp1 + 251.0*u);
        u = unp1;
        unp1 = unp2;
        unp2 = unp3;
        unp3 = unp4;
        unp4 = unp5;
        double mag = cabs(unp5);
        if (mag > tolsup) return -1;
        if (mag < tolinf) return tid;
    }
    return -1;
}

static inline int is_stable_am2(double complex z,
                                           double tolsup, double tolinf, int tid) {
    double complex u = 1.0;
    double complex unp1 = u/(1-z);
    double complex unp2 = unp1*((1 + z/2)/(1 - z/2));
    for (int i = 0; i < 3000; i++) {
        double complex unp3 = (unp2 + (8.0*z/12.0)*unp2 - (z/12.0)*unp1)/(1 - (5.0*z/12.0));
        u = unp1;
        unp1 = unp2;
        unp2 = unp3;

        double mag = cabs(unp3);
        if (mag > tolsup) return -1;
        if (mag < tolinf) return tid;
    }
    return -1;
}

static inline int is_stable_am3(double complex z,
                                           double tolsup, double tolinf, int tid) {
    double complex u = 1.0;
    double complex unp1 = u/(1-z);
    double complex unp2 = unp1*((1 + z/2)/(1 - z/2));
    double complex unp3 = (unp2 + (8.0*z/12.0)*unp2 - (z/12.0)*unp1)/(1 - (5.0*z/12.0));
    for (int i = 0; i < 3000; i++) {
        double complex unp4 = (unp3 + (19.0*z/24.0)*unp3 - (5.0*z/24.0)*unp2 + (z/24.0)*unp1)/(1 - (9.0*z/24.0));
        u = unp1;
        unp1 = unp2;
        unp2 = unp3;
        unp3 = unp4;
        double mag = cabs(unp3);
        if (mag > tolsup) return -1;
        if (mag < tolinf) return tid;
    }
    return -1;
}
static inline int is_stable_am4(double complex z,
                                           double tolsup, double tolinf, int tid) {
    double complex u = 1.0;
    double complex unp1 = u/(1-z);
    double complex unp2 = unp1*((1 + z/2)/(1 - z/2));
    double complex unp3 = (unp2 + (8.0*z/12.0)*unp2 - (z/12.0)*unp1)/(1 - (5.0*z/12.0));
    double complex unp4 = (unp3 + (19.0*z/24.0)*unp3 - (5.0*z/24.0)*unp2 + (z/24.0)*unp1)/(1 - (9.0*z/24.0));
    for (int i = 0; i < 3000; i++) {
        double complex unp5 = (unp4 + (251.0*z/720.0)*unp4 - (1274.0*z/720.0)*unp3 + (2616.0*z/720.0)*unp2 - (2774.0*z/720.0)*unp1)/(1 - (1901.0*z/720.0));
        u = unp1;
        unp1 = unp2;
        unp2 = unp3;
        unp3 = unp4;
        unp4 = unp5;
        double mag = cabs(unp5);
        if (mag > tolsup) return -1;
        if (mag < tolinf) return tid;
    }
    return -1;
}
static inline int is_stable_am5(double complex z,
                                           double tolsup, double tolinf, int tid) {
    double complex u = 1.0;
    double complex unp1 = u/(1-z);
    double complex unp2 = unp1*((1 + z/2)/(1 - z/2));
    double complex unp3 = (unp2 + (8.0*z/12.0)*unp2 - (z/12.0)*unp1)/(1 - (5.0*z/12.0));
    double complex unp4 = (unp3 + (19.0*z/24.0)*unp3 - (5.0*z/24.0)*unp2 + (z/24.0)*unp1)/(1 - (9.0*z/24.0));
    double complex unp5 = (unp4 + (251.0*z/720.0)*unp4 - (1274.0*z/720.0)*unp3 + (2616.0*z/720.0)*unp2 - (2774.0*z/720.0)*unp1)/(1 - (1901.0*z/720.0));
    for (int i = 0; i < 3000; i++) {
        double complex unp6 = (unp5 + (475.0*z/144.0)*unp5 - (1775.0*z/144.0)*unp4 + (3100.0*z/144.0)*unp3 - (3025.0*z/144.0)*unp2 + (251.0*z/144.0)*unp1)/(1 - (1901.0*z/720.0));
        u = unp1;
        unp1 = unp2;
        unp2 = unp3;
        unp3 = unp4;
        unp4 = unp5;
        unp5 = unp6;
        double mag = cabs(unp6);
        if (mag > tolsup) return -1;
        if (mag < tolinf) return tid;
    }
    return -1;
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
// =======================================================
// Salvar um .csv
// =======================================================
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

// =======================================================
// Main 
// =======================================================
int main() {
    double h = 0.01;
    double x_min = -5.0, x_max = 5.0;
    double y_min = -5.0, y_max = 5.0;
    int Nx = (int)((x_max - x_min)/h) + 1;
    int Ny = (int)((y_max - y_min)/h) + 1;
    long int ini = clock();
    //plot("stability_euler", Nx, Ny, x_min, y_min, h, 0);
    //plot("stability_euler_bruteforce",Nx, Ny, x_min, y_min, h, 1);
    //plot("stability_rk2", Nx, Ny, x_min, y_min, h, 2);
    //plot("stability_rk3", Nx, Ny, x_min, y_min, h, 3);
    //plot("stability_rk4", Nx, Ny, x_min, y_min, h, 4);
    //plot("stability_rk4_bruteforce",Nx, Ny, x_min, y_min, h, 5 );
    //plot("stability_rk3_bruteforce",Nx, Ny, x_min, y_min, h, 6 );
    //plot("stability_eulerimplicit_bruteforce",Nx, Ny, x_min, y_min, h, 7 );
    //plot("stability_trapezoid",Nx, Ny, x_min, y_min, h, 8 );
    //plot("stability_bdf2",Nx, Ny, x_min, y_min, h, 9 );
    //plot("stability_bdf3",Nx, Ny, x_min, y_min, h, 10 );
    //plot("stability_ab4",Nx, Ny, x_min, y_min, h, 11 );
    //plot("stability_ab5",Nx, Ny, x_min, y_min, h, 12 );
    //plot("stability_am2",Nx, Ny, x_min, y_min, h, 13 );
    //plot("stability_am3",Nx, Ny, x_min, y_min, h, 14 );
    //plot("stability_am4",Nx, Ny, x_min, y_min, h, 15 );
    plot("stability_am5",Nx, Ny, x_min, y_min, h, 16 );
    long int end = clock();

    printf("%ld", end-ini);
    return 0;
}
