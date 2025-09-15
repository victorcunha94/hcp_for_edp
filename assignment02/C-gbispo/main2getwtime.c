#include <stdio.h>
#include <time.h>
#include <omp.h>
#include "plotagem.h"
// =======================================================
// Main 
// =======================================================
int main() {
    double start, end;
    double x_min = -5.0, x_max = 5.0;
    double y_min = -5.0, y_max = 5.0;
    double h = 0.01;
    int Nx = (int)((x_max - x_min)/h) + 1;
    int Ny = (int)((y_max - y_min)/h) + 1;
    for (int i = 1; i <= omp_get_num_procs(); i++) {
        //omp_set_num_threads(i);
        printf("##########################################\nNúmero de threads=%d\n",i);
        start = omp_get_wtime();
        plot("stability_euler", Nx, Ny, x_min, y_min, h, 0,i);
        end = omp_get_wtime();
        printf("tempo euler%ld\n", end-start);
        start = omp_get_wtime();
        plot("stability_euler_bruteforce",Nx, Ny, x_min, y_min, h, 1,i);
        end = omp_get_wtime();
        printf("tempo euler%ld\n", end-start);
        start = omp_get_wtime();
        plot("stability_rk2", Nx, Ny, x_min, y_min, h, 2,i);
        end = omp_get_wtime();
        printf("tempo range kutta 2º%ld\n", end-start);
        start = omp_get_wtime();
        plot("stability_rk3", Nx, Ny, x_min, y_min, h, 3,i);
        end = omp_get_wtime();
        printf("tempo range kutta 3º%ld\n", end-start);
        start = omp_get_wtime();
        plot("stability_rk4", Nx, Ny, x_min, y_min, h, 4,i);
        end = omp_get_wtime();
        printf("tempo range kutta 4º%ld\n", end-start);
        //plot("stability_rk4_bruteforce",Nx, Ny, x_min, y_min, h, 5 );
        //plot("stability_rk3_bruteforce",Nx, Ny, x_min, y_min, h, 6 );
        start = omp_get_wtime();
        plot("stability_eulerimplicit_bruteforce",Nx, Ny, x_min, y_min, h, 7 ,i);
        end = omp_get_wtime();
        printf("tempo euler implicito%ld\n", end-start);
        start = omp_get_wtime();
        plot("stability_trapezoid",Nx, Ny, x_min, y_min, h, 8 ,i);
        end = omp_get_wtime();
        printf("tempo trapézio%ld\n", end-start);
        start = omp_get_wtime();
        plot("stability_bdf2",Nx, Ny, x_min, y_min, h, 9,i );
        end = omp_get_wtime();
        printf("tempo bdf%ld\n", end-start);
        start = omp_get_wtime();
        plot("stability_bdf3",Nx, Ny, x_min, y_min, h, 10 ,i);
        end = omp_get_wtime();
        printf("tempo bdf3%ld\n", end-start);
        start = omp_get_wtime();
        plot("stability_ab4",Nx, Ny, x_min, y_min, h, 11 ,i);
        end = omp_get_wtime();
        printf("tempo ab4%ld\n", end-start);
        start = omp_get_wtime();
        plot("stability_ab5",Nx, Ny, x_min, y_min, h, 12 ,i);
        end = omp_get_wtime();
        printf("tempo ab5%ld\n", end-start);
        start = omp_get_wtime();
        plot("stability_am2",Nx, Ny, x_min, y_min, h, 13 ,i);
        end = omp_get_wtime();
        printf("tempo am2%ld\n", end-start);
        start = omp_get_wtime();
        plot("stability_am3",Nx, Ny, x_min, y_min, h, 14 ,i);
        end = omp_get_wtime();
        printf("tempo am3%ld\n", end-start);
        start = omp_get_wtime();
        plot("stability_am4",Nx, Ny, x_min, y_min, h, 15 ,i);
        end = omp_get_wtime();
        printf("tempo am4%ld\n", end-start);
        start = omp_get_wtime();
        plot("stability_am5",Nx, Ny, x_min, y_min, h, 16 ,i);
        end = omp_get_wtime();
        printf("tempo am5%ld\n", end-start);
    }
    return 0;
}
