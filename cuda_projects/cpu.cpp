#include <cstdio>

const int N = 1'000'000;
int a[N], b[N], c[N];

void add(int* x, int* y, int* z) {
    for (int i = 0; i < N; ++i)
        z[i] = x[i] + y[i];
}

int main() {
    for (int i = 0; i < N; ++i) a[i] = i, b[i] = 2*i;

    add(a, b, c);

    for (int i = 0; i < 5; ++i)
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
}