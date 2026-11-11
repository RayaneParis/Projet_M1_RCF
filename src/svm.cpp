// mehdi

#include "svm.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>

int predictSVM(const std::vector<Sample>& training, const std::vector<double>& query,
               const std::string& method, double C) {
    if (training.empty() || query.empty()) {
        throw std::invalid_argument("svm requires training samples and a nonempty query");
    }
    if (!std::isfinite(C) || C <= 0.0) {
        throw std::invalid_argument("svm requires a finite, positive C");
    }

    int n = static_cast<int>(training.size());
    int dim = static_cast<int>(query.size());
    std::vector<double> mean(dim, 0.0);
    std::vector<double> sd(dim, 0.0);
    bool differentClasses = false;

    // check the samples and calculate the mean from training data only
    for (int i = 0; i < n; ++i) {
        auto feature = training[i].features.find(method);
        if (feature == training[i].features.end() || feature->second.size() != query.size()) {
            throw std::invalid_argument("svm descriptor is missing or has the wrong size");
        }
        if (training[i].label != training[0].label) differentClasses = true;

        const auto& x = feature->second;
        for (int j = 0; j < dim; ++j) {
            if (!std::isfinite(x[j])) {
                throw std::invalid_argument("svm requires finite training values");
            }
            mean[j] += (x[j] - mean[j]) / (i + 1);
        }
    }
    if (!differentClasses) {
        throw std::invalid_argument("svm requires at least two training classes");
    }

    // calculate how much each feature varies in the training data
    for (const Sample& sample : training) {
        const auto& x = sample.features.at(method);
        for (int j = 0; j < dim; ++j) {
            double difference = x[j] - mean[j];
            sd[j] += difference * difference;
        }
    }
    for (int j = 0; j < dim; ++j) {
        sd[j] = std::sqrt(sd[j] / n);
        if (!std::isfinite(mean[j]) || !std::isfinite(sd[j])) {
            throw std::invalid_argument("svm feature values are too large");
        }
        if (sd[j] == 0.0) sd[j] = 1.0;  // avoid division by zero for constant features
    }

    // opencv stores one sample per row, with float features and integer labels
    cv::Mat samples(n, dim, CV_32F);
    cv::Mat labels(n, 1, CV_32S);
    for (int i = 0; i < n; ++i) {
        const auto& x = training[i].features.at(method);
        for (int j = 0; j < dim; ++j) {
            samples.at<float>(i, j) = static_cast<float>((x[j] - mean[j]) / sd[j]);
        }
        labels.at<int>(i, 0) = training[i].label;
    }

    // scale the query using the same training mean and standard deviation
    cv::Mat test(1, dim, CV_32F);
    for (int j = 0; j < dim; ++j) {
        double value = (query[j] - mean[j]) / sd[j];
        if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max()) {
            throw std::invalid_argument("svm query contains an invalid or too large value");
        }
        test.at<float>(0, j) = static_cast<float>(value);
    }

    // create an rbf svm with fixed parameters
    cv::Ptr<cv::ml::SVM> model = cv::ml::SVM::create();
    model->setType(cv::ml::SVM::C_SVC);
    model->setKernel(cv::ml::SVM::RBF);
    model->setC(C);
    model->setGamma(1.0 / dim);
    model->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,
                                           100000, 0.001));

    // learn from the training samples, then predict the query's class
    if (!model->train(samples, cv::ml::ROW_SAMPLE, labels)) {
        throw std::runtime_error("svm training failed");
    }
    return static_cast<int>(model->predict(test));
}
