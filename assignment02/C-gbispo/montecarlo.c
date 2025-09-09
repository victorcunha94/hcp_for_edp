#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <time.h>

// ======================================
// Fatores de amplificacao: R(z)
// ======================================
static inline double complex euler_amp(double complex z) {
    return 1.0 + z;
}

static inline double complex rk2_amp(double complex z) {
    return 1.0 + z + (cpow(z,2))/2.0;
}

static inline double complex rk3_amp(double complex z) {
    return 1.0 + z + (cpow(z,2))/2.0 + (cpow(z,3))/6.0;
}

static inline double complex rk4_amp(double complex z) {
    return 1.0 + z + (cpow(z,2))/2.0 + (cpow(z,3))/6.0 + (cpow(z,4))/24.0;
}

// Radau IIA (3-stage, 5th order)
static inline double complex radau5_amp(double complex z) {
    static const double A[3][3] = {
        {0.19681547722366, -0.06553542585020, 0.02377097434822},
        {0.39442431473909,  0.29207341166523, -0.04154875212600},
        {0.37640306270047,  0.51248582618842,  0.11111111111111}
    };
    static const double b[3] = {
        0.37640306270047,
        0.51248582618842,
        0.11111111111111
    };

    // Build M = I - z*A
    double complex M[3][3], rhs[3], sol[3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            M[i][j] = (i == j ? 1.0 : 0.0) - z * A[i][j];
        }
        rhs[i] = 1.0; // vector of ones
    }

    // Solve M * sol = rhs (Gaussian elimination, small system)
    for (int i = 0; i < 3; i++) {
        double complex pivot = M[i][i];
        for (int j = i; j < 3; j++) M[i][j] /= pivot;
        rhs[i] /= pivot;
        for (int r = i+1; r < 3; r++) {
            double complex factor = M[r][i];
            for (int j = i; j < 3; j++) M[r][j] -= factor * M[i][j];
            rhs[r] -= factor * rhs[i];
        }
    }
    for (int i = 2; i >= 0; i--) {
        double complex sum = rhs[i];
        for (int j = i+1; j < 3; j++) sum -= M[i][j] * sol[j];
        sol[i] = sum;
    }

    // Compute R(z) = 1 + z * b^T * sol
    double complex bz = 0.0;
    for (int i = 0; i < 3; i++) bz += b[i] * sol[i];
    return 1.0 + z * bz;
}

// ======================================
// Monte Carlo
// ======================================
void mc_border(const char *filename,
                            int N, int Nx, int Ny,
                            double x_min, double x_max,
                            double y_min, double y_max,
                            int method, double eps) {
    unsigned char *image = malloc(Nx * Ny * 3);
    for (int p = 0; p < Nx*Ny; p++) {
        image[3*p]   = 255;
        image[3*p+1] = 255;
        image[3*p+2] = 255;
    }
    if (!image) { perror("malloc failed"); exit(1); }
     
    #pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < N; k++) {
        double x = x_min + (x_max - x_min) * ((double) rand() / RAND_MAX);
        double y = y_min + (y_max - y_min) * ((double) rand() / RAND_MAX);
        double complex z = x + I*y;

        double complex R;
        switch (method) {
            case 0: R = euler_amp(z); break;
            case 1: R = rk2_amp(z);   break;
            case 2: R = rk3_amp(z);   break;
            case 3: R = rk4_amp(z);   break;
            case 4: R = radau5_amp(z); break;
            default: R = euler_amp(z);
        }

        double mag = cabs(R);
        if (fabs(mag - 1.0) < eps) {
            int i = (int)((x - x_min) / (x_max - x_min) * (Nx - 1));
            int j = (int)((y - y_min) / (y_max - y_min) * (Ny - 1));
            if (i >= 0 && i < Nx && j >= 0 && j < Ny) {
                int idx = (j * Nx + i) * 3;
                image[idx]   = 0; 
                image[idx+1] = 0;
                image[idx+2] = 0;
            }
        }
    }

    FILE *f = fopen(filename, "wb");
    if (!f) { perror("fopen failed"); exit(1); }
    fprintf(f, "P6\n%d %d\n255\n", Nx, Ny);
    fwrite(image, 1, Nx * Ny * 3, f);
    fclose(f);

    free(image);
}

// ======================================
// Main
// ======================================
int main() {
    srand(time(NULL));

    int N = 5000000;
    double h = 0.01;
    double x_min = -10.0, x_max = 10.0;
    double y_min = -10.0, y_max = 10.0;
    int Nx = (int)((x_max - x_min)/h) + 1;
    int Ny = (int)((y_max - y_min)/h) + 1;
    double eps = 0.01;  

    long int ini = clock();
    mc_border("mc_euler.ppm", N, Nx, Ny, x_min, x_max, y_min, y_max, 0, eps);
    mc_border("mc_rk2.ppm",   N, Nx, Ny, x_min, x_max, y_min, y_max, 1, eps);
    mc_border("mc_rk3.ppm",   N, Nx, Ny, x_min, x_max, y_min, y_max, 2, eps);
    mc_border("mc_rk4.ppm",   N, Nx, Ny, x_min, x_max, y_min, y_max, 3, eps);
    mc_border("mc_radau5.ppm",N, Nx, Ny, x_min, x_max, y_min, y_max, 4, eps);
    long int end = clock();

    printf("Elapsed: %ld ticks\n", end-ini);
    return 0;
}
