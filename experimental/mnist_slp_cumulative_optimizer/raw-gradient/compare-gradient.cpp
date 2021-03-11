#include <cstdio>
#include <iostream>
#include <map>
#include <vector>

void read_gradient_f32(const char *folder, int idx, int count,
                       std::vector<float> &x)
{
    char filename[1024];
    sprintf(filename, "%s/%06d-%s-%d.data", folder, idx, "f32", count);
    x.resize(count);
    FILE *fp = fopen(filename, "rb");
    fread(x.data(), sizeof(float), x.size(), fp);
    fclose(fp);
}

int main()
{
    int count std::vector<float> g1, g2, g;
    read_gradient_f32("dev-bs=100", 0, 10, g1);
    read_gradient_f32("dev-bs=100", 1, 10, g2);
    return 0;
}
