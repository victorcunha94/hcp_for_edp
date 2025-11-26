#include <stdio.h>
#include <stdlib.h>

void func1(void)
{

 int x;

 x = 10;

}

void f(void)
{
  int t;
  scanf("%d", &t);
  
  if(t == 1) {

    char s[80]; /* isto é criado apenas na entrada deste bloco */
    printf("entre com o nome:");
    fgets(s, 80, NULL);
    /* faz alguma coisa ...*/
  }

}
