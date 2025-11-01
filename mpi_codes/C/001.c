#include <stdio.h>


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
    gets(s);
    /* faz alguma coisa ...*/
  }

}
