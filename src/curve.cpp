// Rayane — classification precision/recall
#include "curve.hpp"

#include <algorithm>
#include <stdexcept>

PRCurve precisionRecallCurve(const std::vector<int>& truth,
                            const std::vector<std::vector<int>>& votes, int positiveClass) {
    if (truth.size() < 2 || votes.size() != truth.size() || votes.front().empty()) {
        throw std::invalid_argument("precision/recall requires votes for at least two labeled images");
    }
    const auto totalPositive = std::count(truth.begin(), truth.end(), positiveClass);
    if (totalPositive == 0) {
        throw std::invalid_argument("precision/recall requires a positive example for the class");
    }

    // All images use the same classifier (12 neighbors, or five descriptor votes).
    const std::size_t votesPerImage = votes.front().size();
    struct Bin {
        std::size_t images = 0;
        std::size_t positives = 0;
    };
    std::vector<Bin> bins(votesPerImage + 1);
    for (std::size_t i = 0; i < truth.size(); ++i) {
        if (votes[i].size() != votesPerImage) {
            throw std::invalid_argument("precision/recall requires the same number of votes per image");
        }
        auto& bin = bins[std::count(votes[i].begin(), votes[i].end(), positiveClass)];
        ++bin.images;
        if (truth[i] == positiveClass) ++bin.positives;
    }

    PRCurve curve;
    curve.points.push_back({std::nullopt, 0.0, 1.0});
    std::size_t selected = 0, truePositive = 0;
    // Lower the threshold one distinct score at a time, admitting tied images together.
    for (std::size_t level = bins.size(); level > 0; --level) {
        const std::size_t count = level - 1;
        const Bin& bin = bins[count];
        if (bin.images == 0) continue;
        selected += bin.images;
        truePositive += bin.positives;
        const double recall = static_cast<double>(truePositive) / totalPositive;
        const double precision = static_cast<double>(truePositive) / selected;
        const PRPoint& previous = curve.points.back();
        curve.auc += (recall - previous.recall) * (precision + previous.precision) / 2.0;
        curve.points.push_back({static_cast<double>(count) / votesPerImage, recall, precision});
    }
    return curve;
}
