//
// Created by bruno on 07/09/25.
//

#include "metodos_numericos.h"
int is_stable(double complex z) {

    return cabs(z) <= 1.0;
}
// Euler
double complex euler_amp(double complex z) {
    return 1.0 + z;
}


// RK2
double complex rk2_amp(double complex z) {
    return 1.0 + z + (cpow(z,2))/2.0;
}

// RK3
double complex rk3_amp(double complex z) {
    return 1.0 + z + (cpow(z,2))/2.0 + (cpow(z,3))/6.0;
}


// RK4
double complex rk4_amp(double complex z) {
    return 1.0 + z + (cpow(z,2))/2.0 + (cpow(z,3))/6.0 + (cpow(z,4))/24.0;
}
// =======================================================
// Forca Bruta
// =======================================================

// Euler forca bruta
int is_stable_euler_bruteforce(double complex z,
                                             double tolsup, double tolinf) {
    double complex u = 1.0;
    for (int i = 0; i < 3000; i++) {
        u = (1.0 + z) * u;
        double mag = cabs(u);
        if (mag > tolsup) return 0; // instavel
        if (mag < tolinf) return 1; // estavel
    }
    return 0;
}
int is_stable_eulerimplicit_bruteforce(double complex z,
                                             double tolsup, double tolinf, int tid) {
    double complex u = 1.0;
    for (int i = 0; i < 3000; i++) {
        u = u/(1.0 - z);
        double mag = cabs(u);
        if (mag > tolsup) return -1; // instavel
        if (mag < tolinf) return tid; // estavel
    }
    return -1;
}

int is_stable_rk3_bruteforce(double complex z,
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

int is_stable_rk4_bruteforce(double complex z,
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

int is_stable_trapez(double complex z,
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

int is_stable_ab2(double complex z,
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

int is_stable_ab3(double complex z,
                                           double tolsup, double tolinf, int tid) {
    double complex u = 1.0;
    double complex unp1 = (1+z)*u;
    double complex unp2 = unp1 + (z/2)*(3.0*unp1 - u);
    for (int i = 0; i < 3000; i++) {
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
int is_stable_ab4(double complex z,
                                           double tolsup, double tolinf, int tid) {
    double complex u = 1.0;
    double complex unp1 = (1+z)*u;
    double complex unp2 = unp1 + (z/2)*(3.0*unp1 - u);
    double complex unp3 = unp2 + (z/12)*(23.0*unp2 - 16.0*unp1 + 5.0*u);
    for (int i = 0; i < 3000; i++) {
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
int is_stable_ab5(double complex z,
                                           double tolsup, double tolinf, int tid) {
    double complex u = 1.0;
    double complex unp1 = (1+z)*u;
    double complex unp2 = unp1 + (z/2)*(3.0*unp1 - u);
    double complex unp3 = unp2 + (z/12)*(23.0*unp2 - 16.0*unp1 + 5.0*u);
    double complex unp4 = unp3 + (z/24)*(55.0*unp3 - 59.0*unp2 + 37.0*unp1 - 9.0*u);
    for (int i = 0; i < 3000; i++) {
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

int is_stable_am2(double complex z,
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
int is_stable_am3(double complex z,
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
int is_stable_am4(double complex z,
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
int is_stable_am5(double complex z,
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