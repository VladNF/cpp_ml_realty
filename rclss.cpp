#include "data.hpp"
#include <filesystem>
#include <set>

void show_relevant_records(
        data::sample_type &s,
        decltype(data::make_model()) &classifier,
        decltype(data::make_normalizer()) &normalizer,
        std::string &model_path
) {
    namespace fs = std::filesystem;
    auto predicted_cluster = classifier(normalizer(s));
    auto class_path = model_path + "." + std::to_string(predicted_cluster);
    if (!fs::exists(class_path)) {
        std::cout << "No relevant records found...\n";
        return;
    }
    
    std::ifstream class_file(class_path);
    auto cmp = [&](const data::sample_type &lhs, const data::sample_type &rhs) {
        auto r1 = (lhs(0)  - s(0)) * (lhs(0)  - s(0)) + (lhs(1)  - s(1)) * (lhs(1)  - s(1));
        auto r2 = (rhs(0)  - s(0)) * (rhs(0)  - s(0)) + (rhs(1)  - s(1)) * (rhs(1)  - s(1));
        return r1 < r2;
    };
    std::multiset<data::sample_type, decltype(cmp)> relevant_records(cmp);

    data::sample_type row;
    while (class_file.peek() != EOF) {
        data::read_sample(class_file, row, ',');
        relevant_records.insert(row);
    }

    for (auto &r : relevant_records) {
        std::cout << dlib::csv << dlib::trans(r);
    }

    std::cout << std::endl;
}

int main(int argc, char const *argv[]) {
    const auto usage = "rclss <model path>";
    std::string model_path;
    try {
        if (argc < 2) {
            std::cerr << "Not all required arguments were passed. Program usage:\n" << usage;
            exit(1);
        }
        model_path = argv[1];
    } catch (const std::exception &e) {
        std::cerr << "Invalid argument passed: " << e.what();
        exit(1);
    }

    auto classifier = data::make_model();
    dlib::vector_normalizer<data::sample_type> normalizer;

    dlib::deserialize(model_path + ".mod") >> classifier;
    dlib::deserialize(model_path + ".nrm") >> normalizer;

    // Read & find relevant records
    data::sample_type sample;
    while (std::cin.peek() != EOF) {
        data::read_sample(std::cin, sample);
        show_relevant_records(sample, classifier, normalizer, model_path);
    }
}
