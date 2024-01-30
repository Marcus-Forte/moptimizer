#pragma once

namespace util
{
    std::vector<double> linspace(double min, double max, size_t num_elements)
    {
        double step = (max - min) / num_elements;
        std::vector<double> ret;
        ret.reserve(num_elements);
        for (int i = 0; i < num_elements; ++i)
        {
            ret.emplace_back(min + step * i);
        }
        return ret;
    }
}