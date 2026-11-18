// Rayane 3.c Curve precision/recall
#pragma once

#include <string>
#include <vector>
#include "dataset.hpp"
#include "distance.hpp"

struct PRPoint {
    int rank = 0;
    double recall = 0.0;
    double precision = 0.0;
};

struct PRCurve {
    std::vector<PRPoint> points;
    double auc = 0.0;
};

// 2.a 
PRCurve precisionRecallCurve(const std::vector<Sample>& data, 
                             const std::string& method, 
                             DistanceType type, int K);

// 2.c 
PRCurve precisionRecallCurveVote(const std::vector<Sample>& data, 
                                 const std::vector<std::string>& methods, 
                                 DistanceType type, int K);

