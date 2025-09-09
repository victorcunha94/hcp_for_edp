//
// Created by bruno on 07/09/25.
//

#ifndef METODOS_NUMERICOS_H

#define METODOS_NUMERICOS_H
#include <complex.h>
int is_stable(double complex z);
// =======================================================
// Fatores de amplificacao: R(z)
// =======================================================
double complex euler_amp(double complex z);
double complex rk2_amp(double complex z);
double complex rk3_amp(double complex z);
double complex rk4_amp(double complex z);
int is_stable_euler_bruteforce(double complex z,
                               double tolsup, double tolinf);
int is_stable_eulerimplicit_bruteforce(double complex z,
                               double tolsup, double tolinf, int tid);
int is_stable_rk3_bruteforce(double complex z,
                             double tolsup, double tolinf);
int is_stable_rk4_bruteforce(double complex z,
                             double tolsup, double tolinf, int tid);
int is_stable_trapez(double complex z,
                             double tolsup, double tolinf, int tid);
int is_stable_ab2(double complex z,
                             double tolsup, double tolinf, int tid);
int is_stable_ab3(double complex z,
                             double tolsup, double tolinf, int tid);
int is_stable_ab4(double complex z,
                             double tolsup, double tolinf, int tid);
int is_stable_ab5(double complex z,
                             double tolsup, double tolinf, int tid);
int is_stable_am2(double complex z,
                             double tolsup, double tolinf, int tid);
int is_stable_am3(double complex z,
                             double tolsup, double tolinf, int tid);
int is_stable_am4(double complex z,
                             double tolsup, double tolinf, int tid);
int is_stable_am5(double complex z,
                                    double tolsup, double tolinf, int tid);
#endif //METODOS_NUMERICOS_H
