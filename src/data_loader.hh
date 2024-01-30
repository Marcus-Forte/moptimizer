#pragma once

#include <string>
#include <vector>
#include <fstream>

void loadData(const std::string &path, std::vector<double> &x, std::vector<double> &y)
{
    std::ifstream file(path);

    if (!file.is_open())
        throw std::runtime_error("Unable to open file. Exiting..");

    x.clear();
    y.clear();
    double x_, y_;
    while (file >> y_ >> x_)
    {
        x.push_back(x_);
        y.push_back(y_);
    }
}