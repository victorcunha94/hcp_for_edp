
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <string.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void make_filename(char *dest, size_t size, const char *basename, const char *ext) {
    snprintf(dest, size, "%s%s", basename, ext);
}

static inline int is_stable_euler_bruteforce(double complex z, double tolsup, double tolinf, int tid) {
    double complex u = 1.0;
    for (int i = 0; i < 3000; ++i) {
        u = (1.0 + z) * u;
        double mag = cabs(u);
        if (mag > tolsup) return -1;
        if (mag < tolinf) return tid;
    }
    return -1;
}

static inline int is_stable_rk2_bruteforce(double complex z, double tolsup, double tolinf, int tid) {
    // explicit midpoint / Heun-like 2-stage (classical RK2)
    double complex u = 1.0;
    for (int i = 0; i < 3000; ++i) {
        double complex k1 = z * u;
        double complex k2 = z * (u + 0.5 * k1);
        u = u + k2;
        double mag = cabs(u);
        if (mag > tolsup) return -1;
        if (mag < tolinf) return tid;
    }
    return -1;
}

static inline int is_stable_rk3_bruteforce(double complex z, double tolsup, double tolinf, int tid) {
    double complex u = 1.0;
    for (int i = 0; i < 3000; ++i) {
        double complex k1 = z * u;
        double complex k2 = z * (u + k1 / 3.0);
        double complex k3 = z * (u + 2.0 * k2 / 3.0);
        u = u + (k1 + 3.0 * k3) / 4.0;
        double mag = cabs(u);
        if (mag > tolsup) return -1;
        if (mag < tolinf) return tid;
    }
    return -1;
}

static inline int is_stable_rk4_bruteforce(double complex z, double tolsup, double tolinf, int tid) {
    double complex u = 1.0;
    for (int i = 0; i < 3000; ++i) {
        double complex k1 = z * u;
        double complex k2 = z * (u + 0.5 * k1);
        double complex k3 = z * (u + 0.5 * k2);
        double complex k4 = z * (u + k3);
        u = u + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
        double mag = cabs(u);
        if (mag > tolsup) return -1;
        if (mag < tolinf) return tid;
    }
    return -1;
}

static inline int is_stable_trapez_bruteforce(double complex z, double tolsup, double tolinf, int tid) {
    double complex u = 1.0;
    double complex R = (1.0 + z/2.0) / (1.0 - z/2.0);
    for (int i = 0; i < 3000; ++i) {
        u *= R;
        double mag = cabs(u);
        if (mag > tolsup) return -1;
        if (mag < tolinf) return tid;
    }
    return -1;
}

static inline int is_stable_ieuler_bruteforce(double complex z, double tolsup, double tolinf, int tid) {
    double complex u = 1.0;
    double complex R = 1.0 / (1.0 - z);
    for (int i = 0; i < 3000; ++i) {
        u *= R;
        double mag = cabs(u);
        if (mag > tolsup) return -1;
        if (mag < tolinf) return tid;
    }
    return -1;
}

static inline int is_stable_bdf2_bruteforce(double complex z, double tolsup, double tolinf, int tid) {
    double complex u_nm1 = 1.0, u_n = 1.0;
    for (int i=0;i<3000;++i) {
        double complex numerator = 2.0*u_n - 0.5*u_nm1;
        double complex denom = 1.5 - z;
        double complex u_np1 = numerator / denom;
        u_nm1 = u_n; u_n = u_np1;
        double mag = cabs(u_n);
        if (mag > tolsup) return -1;
        if (mag < tolinf) return tid;
    }
    return -1;
}

static inline int is_stable_trbdf2_bruteforce(double complex z, double tolsup, double tolinf, int tid) {
    double complex u = 1.0;
    const double gamma = 2.0 - sqrt(2.0);
    for (int i=0;i<3000;++i) {
        double complex z1 = z * gamma;
        double complex u_star = ((1.0 + z1/2.0) / (1.0 - z1/2.0)) * u; // trapezoid on gamma portion
        double complex z2 = z * (1.0 - gamma);
        double complex u_next = u_star / (1.0 - z2); // implicit Euler on remaining fraction
        u = u_next;
        double mag = cabs(u);
        if (mag > tolsup) return -1;
        if (mag < tolinf) return tid;
    }
    return -1;
}

static inline int is_stable_ab2_bruteforce(double complex z, double tolsup, double tolinf, int tid) {
    double complex u_nm1 = 1.0, u_n = 1.0;
    for (int i=0;i<3000;++i) {
        double complex u_np1 = (1.0 + 1.5*z) * u_n - (0.5*z) * u_nm1;
        u_nm1 = u_n; u_n = u_np1;
        double mag = cabs(u_n);
        if (mag > tolsup) return -1;
        if (mag < tolinf) return tid;
    }
    return -1;
}

static inline int is_stable_ab3_bruteforce(double complex z, double tolsup, double tolinf, int tid) {
    double complex u_nm2 = 1.0, u_nm1 = 1.0, u_n = 1.0;
    for (int i=0;i<3000;++i) {
        double complex u_np1 = u_n + z*( (23.0/12.0)*u_n - (16.0/12.0)*u_nm1 + (5.0/12.0)*u_nm2 );
        u_nm2 = u_nm1; u_nm1 = u_n; u_n = u_np1;
        double mag = cabs(u_n);
        if (mag > tolsup) return -1;
        if (mag < tolinf) return tid;
    }
    return -1;
}

static inline int is_stable_ab4_bruteforce(double complex z, double tolsup, double tolinf, int tid) {
    double complex u_nm3 = 1.0, u_nm2 = 1.0, u_nm1 = 1.0, u_n = 1.0;
    for (int i=0;i<3000;++i) {
        double complex u_np1 = u_n + z*( (55.0/24.0)*u_n - (59.0/24.0)*u_nm1 + (37.0/24.0)*u_nm2 - (9.0/24.0)*u_nm3 );
        u_nm3 = u_nm2; u_nm2 = u_nm1; u_nm1 = u_n; u_n = u_np1;
        double mag = cabs(u_n);
        if (mag > tolsup) return -1;
        if (mag < tolinf) return tid;
    }
    return -1;
}

static inline int is_stable_am2_bruteforce(double complex z, double tolsup, double tolinf, int tid) {
    double complex u = 1.0;
    double complex R = (1.0 + z/2.0) / (1.0 - z/2.0);
    for (int i=0;i<3000;++i) {
        u *= R;
        double mag = cabs(u);
        if (mag > tolsup) return -1;
        if (mag < tolinf) return tid;
    }
    return -1;
}

static inline int is_stable_am3_bruteforce(double complex z, double tolsup, double tolinf, int tid) {
    double complex u_nm1 = 1.0, u_n = 1.0;
    for (int i=0;i<3000;++i) {
        double complex denom = 1.0 - (5.0/12.0)*z;
        double complex numer = (1.0 + (2.0/3.0)*z) * u_n - (z/12.0) * u_nm1;
        double complex u_np1 = numer / denom;
        u_nm1 = u_n; u_n = u_np1;
        double mag = cabs(u_n);
        if (mag > tolsup) return -1;
        if (mag < tolinf) return tid;
    }
    return -1;
}

static inline int is_stable_am4_bruteforce(double complex z, double tolsup, double tolinf, int tid) {
    double complex u_nm2 = 1.0, u_nm1 = 1.0, u_n = 1.0;
    for (int i=0;i<3000;++i) {
        double complex denom = 1.0 - (9.0/24.0) * z;
        double complex numer = (1.0 + (19.0/24.0) * z) * u_n + z * ( (-5.0/24.0) * u_nm1 + (1.0/24.0) * u_nm2 );
        double complex u_np1 = numer / denom;
        u_nm2 = u_nm1; u_nm1 = u_n; u_n = u_np1;
        double mag = cabs(u_n);
        if (mag > tolsup) return -1;
        if (mag < tolinf) return tid;
    }
    return -1;
}

static inline int is_stable_method_bruteforce(double complex z, int method, double tolsup, double tolinf, int tid) {
    switch(method) {
        case 0: return is_stable_euler_bruteforce(z,tolsup,tolinf,tid);
        case 2: return is_stable_rk2_bruteforce(z,tolsup,tolinf,tid);
        case 3: return is_stable_rk3_bruteforce(z,tolsup,tolinf,tid);
        case 4: return is_stable_rk4_bruteforce(z,tolsup,tolinf,tid);
        case 7: return is_stable_ieuler_bruteforce(z,tolsup,tolinf,tid);
        case 8: return is_stable_trapez_bruteforce(z,tolsup,tolinf,tid);
        case 16: return is_stable_bdf2_bruteforce(z,tolsup,tolinf,tid);
        case 17: return is_stable_trbdf2_bruteforce(z,tolsup,tolinf,tid);
        case 18: return is_stable_ab2_bruteforce(z,tolsup,tolinf,tid);
        case 19: return is_stable_am2_bruteforce(z,tolsup,tolinf,tid);
        case 20: return is_stable_ab3_bruteforce(z,tolsup,tolinf,tid);
        case 21: return is_stable_ab4_bruteforce(z,tolsup,tolinf,tid);
        case 22: return is_stable_am3_bruteforce(z,tolsup,tolinf,tid);
        case 23: return is_stable_am4_bruteforce(z,tolsup,tolinf,tid);
        default: return -1;
    };
};
static inline int is_stable_pece_bruteforce(double complex z,
                                            double tolsup, double tolinf,
                                            int pred_method, int corr_method,
                                            int nCE, int tid) {
    double complex u_nm3 = 1.0, u_nm2 = 1.0, u_nm1 = 1.0, u_n = 1.0;

    for (int step = 0; step < 10000; ++step) {
        // -------------------------------
        // Predictor step
        // -------------------------------
        double complex u_pred;
        switch (pred_method) {
            case 0: // Euler
                u_pred = u_n + z*u_n;
                break;
            case 2: { // RK2
                double complex k1 = z*u_n;
                double complex k2 = z*(u_n + 0.5*k1);
                u_pred = u_n + k2;
                break;
            }
            case 3: { // RK3
                double complex k1 = z*u_n;
                double complex k2 = z*(u_n + k1/3.0);
                double complex k3 = z*(u_n + 2.0*k2/3.0);
                u_pred = u_n + (k1 + 3.0*k3)/4.0;
                break;
            }
            case 4: { // RK4
                double complex k1 = z*u_n;
                double complex k2 = z*(u_n + 0.5*k1);
                double complex k3 = z*(u_n + 0.5*k2);
                double complex k4 = z*(u_n + k3);
                u_pred = u_n + (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0;
                break;
            }
            case 18: // AB2
                u_pred = (1.0 + 1.5*z) * u_n - (0.5*z)*u_nm1;
                break;
            case 20: // AB3
                u_pred = u_n + z*((23.0/12.0)*u_n - (16.0/12.0)*u_nm1 + (5.0/12.0)*u_nm2);
                break;
            case 21: // AB4
                u_pred = u_n + z*((55.0/24.0)*u_n - (59.0/24.0)*u_nm1
                                   + (37.0/24.0)*u_nm2 - (9.0/24.0)*u_nm3);
                break;
            default: // fallback Euler
                u_pred = u_n + z*u_n;
                break;
        }

        // -------------------------------
        // PECE loop
        // -------------------------------
        double complex u_est = u_pred;
        for (int k = 0; k < nCE; ++k) {
            double complex f_est = z*u_est; // <-- EVALUATION step!

            switch (corr_method) {
                case 7: // implicit Euler (fixed-point style)
                    // u_{n+1} = u_n + h f(t_{n+1}, u_{n+1})
                    u_est = u_n + f_est;
                    break;

                case 8: // trapezoid (AM2)
                    // u_{n+1} = u_n + (h/2)(f_n + f_{n+1})
                    u_est = u_n + 0.5*(z*u_n + f_est);
                    break;

                case 16: // BDF2 (iterative form)
                    // (3/2) u_{n+1} - 2 u_n + (1/2) u_{n-1} = h f_{n+1}
                    // => u_{n+1}^{k+1} = (2 u_n - 0.5 u_{n-1} + h f^{(k)}_{n+1}) / 1.5
                    u_est = (2.0*u_n - 0.5*u_nm1 + f_est) / 1.5;
                    break;

                case 19: // AM2 (same as trapezoid, but included separately)
                    u_est = u_n + 0.5*(z*u_n + f_est);
                    break;

                case 22: // AM3
                    // u_{n+1} = u_n + h(5/12 f_{n+1} + 2/3 f_n - 1/12 f_{n-1})
                    u_est = u_n + (5.0/12.0)*f_est + (2.0/3.0)*(z*u_n) - (1.0/12.0)*(z*u_nm1);
                    break;

                case 23: // AM4
                    // u_{n+1} = u_n + h(9/24 f_{n+1} + 19/24 f_n - 5/24 f_{n-1} + 1/24 f_{n-2})
                    u_est = u_n + (9.0/24.0)*f_est + (19.0/24.0)*(z*u_n)
                                 - (5.0/24.0)*(z*u_nm1) + (1.0/24.0)*(z*u_nm2);
                    break;

                default: // no correction
                    u_est = u_pred;
                    break;
            }
        }

        // -------------------------------
        // Shift history and update
        // -------------------------------
        u_nm3 = u_nm2; u_nm2 = u_nm1; u_nm1 = u_n; u_n = u_est;

        // Stability test
        double mag = cabs(u_n);
        if (mag > tolsup) return -1;
        if (mag < tolinf) return tid;
    }

    return -1;
}

// -----------------------------
// IO: save ppm, png, csv
// -----------------------------
void save_ppm(const char *filename, int Nx, int Ny, int *stability,
              double x_min, double x_max, double y_min, double y_max) {
    FILE *f = fopen(filename, "w");
    if (!f) { perror("save_ppm fopen"); return; }
    fprintf(f, "P3\n%d %d\n255\n", Nx, Ny);
    for (int j = Ny-1; j >= 0; --j) {
        for (int i = 0; i < Nx; ++i) {
            int v = stability[i*Ny + j];
            if (v == -1) fprintf(f, "255 255 255 ");
            else fprintf(f, "0 0 0 ");
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

void save_png(const char *filename, int Nx, int Ny, int *stability) {
    unsigned char *img = malloc(3 * Nx * Ny);
    if (!img) { perror("save_png malloc"); return; }
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            // flip vertically so PNG matches PPM orientation
            int v = stability[i*Ny + (Ny-1 - j)];
            unsigned char c = (v == -1) ? 255 : 0;
            int idx = 3 * (j * Nx + i);
            img[idx + 0] = c;
            img[idx + 1] = c;
            img[idx + 2] = c;
        }
    }
    stbi_write_png(filename, Nx, Ny, 3, img, Nx * 3);
    free(img);
}

void save_csv(const char *filename, int Nx, int Ny, int *stability,
              double x_min, double x_max, double y_min, double y_max) {
    FILE *f = fopen(filename, "w");
    if (!f) { perror("save_csv fopen"); return; }
    fprintf(f, "x,y,stable\n");
    double dx = (x_max - x_min) / (Nx - 1);
    double dy = (y_max - y_min) / (Ny - 1);
    for (int i = 0; i < Nx; ++i) {
        double x = x_min + i * dx;
        for (int j = 0; j < Ny; ++j) {
            double y = y_min + j * dy;
            int v = stability[i*Ny + j];
            fprintf(f, "%f,%f,%d\n", x, y, v);
        }
    }
    fclose(f);
}

void plot_method(const char *basename, int Nx, int Ny,
                 double x_min, double y_min, double h,
                 int method, int write_png, int write_ppm) {
    int *stability = malloc(Nx * Ny * sizeof(int));
    if (!stability) { perror("plot_method malloc"); exit(1); }

    double tolsup = 1e6;
    double tolinf = 1e-6;

    #pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < Ny; ++j) {
        double y = y_min + j * h;
        for (int i = 0; i < Nx; ++i) {
            double x = x_min + i * h;
            double complex z = x + I * y;
            int tid = omp_get_thread_num();
            int val = is_stable_method_bruteforce(z, method, tolsup, tolinf, tid);
            stability[i*Ny + j] = val;
        }
    }

    char fname[256];
    if (write_png) { make_filename(fname, sizeof(fname), basename, ".png"); save_png(fname, Nx, Ny, stability); }
    if (write_ppm) { make_filename(fname, sizeof(fname), basename, ".ppm"); save_ppm(fname, Nx, Ny, stability, x_min, x_min + (Nx-1)*h, y_min, y_min + (Ny-1)*h); }
    make_filename(fname, sizeof(fname), basename, ".csv");
    save_csv(fname, Nx, Ny, stability, x_min, x_min + (Nx-1)*h, y_min, y_min + (Ny-1)*h);

    free(stability);
}

void plot_pece(const char *basename, int Nx, int Ny,
               double x_min, double y_min, double h,
               int pred, int corr, int nCE,
               int write_png, int write_ppm) {
    int *stability = malloc(Nx * Ny * sizeof(int));
    if (!stability) { perror("plot_pece malloc"); exit(1); }

    double tolsup = 1e6;
    double tolinf = 1e-6;

    #pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < Ny; ++j) {
        double y = y_min + j * h;
        for (int i = 0; i < Nx; ++i) {
            double x = x_min + i * h;
            double complex z = x + I * y;
            int tid = omp_get_thread_num();
            int val = is_stable_pece_bruteforce(z, tolsup, tolinf, pred, corr, nCE, tid);
            stability[i*Ny + j] = val;
        }
    }

    char fname[256];
    if (write_png) { make_filename(fname, sizeof(fname), basename, ".png"); save_png(fname, Nx, Ny, stability); }
    if (write_ppm) { make_filename(fname, sizeof(fname), basename, ".ppm"); save_ppm(fname, Nx, Ny, stability, x_min, x_min + (Nx-1)*h, y_min, y_min + (Ny-1)*h); }
    make_filename(fname, sizeof(fname), basename, ".csv");
    save_csv(fname, Nx, Ny, stability, x_min, x_min + (Nx-1)*h, y_min, y_min + (Ny-1)*h);

    free(stability);
}

int main(int argc, char **argv) {
    // defaults
    double x_min = -5.0, x_max = 5.0, y_min = -5.0, y_max = 5.0, h = 0.01;
    int write_png = 0, write_ppm = 0;

    while (argc > 1) {
        if (strcmp(argv[argc-1], "--png") == 0) { write_png = 1; argc--; }
        else if (strcmp(argv[argc-1], "--ppm") == 0) { write_ppm = 1; argc--; }
        else break;
    }
    if (!write_png && !write_ppm) write_ppm = 1; 

    if (argc < 2) {
        printf("Usage:\n");
        printf("  %s method output_name [--png] [--ppm]\n", argv[0]);
        printf("  %s method x_min x_max y_min y_max h output_name [--png] [--ppm]\n", argv[0]);
        printf("  %s pece pred corr nCE output_name [--png] [--ppm]\n", argv[0]);
        printf("  %s pece pred corr nCE x_min x_max y_min y_max h output_name [--png] [--ppm]\n", argv[0]);
        printf("\nMethod codes: 0=Euler,2=RK2,3=RK3 ,4=RK4 ,7=IE ,8=Trapezoid \n");
        printf("              16=BDF2,17=TrBDF2,18=AB2,19=AM2,20=AB3,21=AB4,22=AM3,23=AM4\n");
        return 0;
    }

    clock_t t0 = clock();

    if (strcmp(argv[1], "pece") == 0) {
        // two calling styles: short (6 args total) or extended (11 args total)
        if (argc == 6) {
            int pred = atoi(argv[2]);
            int corr = atoi(argv[3]);
            int nCE  = atoi(argv[4]);
            const char *out = argv[5];
            int Nx = (int)((x_max - x_min)/h) + 1;
            int Ny = (int)((y_max - y_min)/h) + 1;
            plot_pece(out, Nx, Ny, x_min, y_min, h, pred, corr, nCE, write_png, write_ppm);
        } else if (argc == 11) {
            int pred = atoi(argv[2]);
            int corr = atoi(argv[3]);
            int nCE  = atoi(argv[4]);
            x_min = atof(argv[5]); x_max = atof(argv[6]);
            y_min = atof(argv[7]); y_max = atof(argv[8]);
            h     = atof(argv[9]);
            const char *out = argv[10];
            if (h <= 0.0) { fprintf(stderr, "h must be > 0\n"); return 1; }
            int Nx = (int)((x_max - x_min)/h) + 1;
            int Ny = (int)((y_max - y_min)/h) + 1;
            plot_pece(out, Nx, Ny, x_min, y_min, h, pred, corr, nCE, write_png, write_ppm);
        } else {
            fprintf(stderr, "Invalid pece invocation. See usage.\n");
            return 1;
        }
    } else {
        // single method: short (3 args) or extended (8 args)
        if (argc == 3) {
            int method = atoi(argv[1]);
            const char *out = argv[2];
            int Nx = (int)((x_max - x_min)/h) + 1;
            int Ny = (int)((y_max - y_min)/h) + 1;
            plot_method(out, Nx, Ny, x_min, y_min, h, method, write_png, write_ppm);
        } else if (argc == 8) {
            int method = atoi(argv[1]);
            x_min = atof(argv[2]); x_max = atof(argv[3]);
            y_min = atof(argv[4]); y_max = atof(argv[5]);
            h     = atof(argv[6]);
            const char *out = argv[7];
            if (h <= 0.0) { fprintf(stderr, "h must be > 0\n"); return 1; }
            int Nx = (int)((x_max - x_min)/h) + 1;
            int Ny = (int)((y_max - y_min)/h) + 1;
            plot_method(out, Nx, Ny, x_min, y_min, h, method, write_png, write_ppm);
        } else {
            fprintf(stderr, "Invalid invocation. See usage.\n");
            return 1;
        }
    }

    clock_t t1 = clock();
    double elapsed = (double)(t1 - t0) / CLOCKS_PER_SEC;
    printf("Elapsed: %.3f s\n", elapsed);
    return 0;
}
