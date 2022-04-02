#include "data.hpp"
#include <vector>

int main(int argc, char const *argv[]) {
    const auto usage = "rclst <# of clusters> <model path>";
    std::string model_path;
    unsigned num_clusters;
    try {
        if (argc < 3) {
            std::cerr << "Not all required arguments were passed. Program usage:\n" << usage;
            exit(1);
        }
        num_clusters = std::stol(argv[1]);
        model_path = argv[2];
    } catch (const std::exception &e) {
        std::cerr << "Invalid argument passed: " << e.what();
        exit(1);
    }


    auto classifier = data::make_model();
    std::vector<data::sample_type> samples;
    std::vector<data::sample_type> normed_samples;
    std::vector<data::sample_type> initial_centers;

    // Read & add features & normalize data
    data::input_type row;
    data::sample_type sample;
    std::cout << "Reading input... " << std::flush;
    while (std::cin.peek() != EOF) {
        data::read_sample(std::cin, row);
        for (int i = 0; i < 6; ++i) sample(i) = row(i);
        sample(6) = row(7) == 1 || row(7) == row(6) ? 0 : 1;
        samples.push_back(sample);
    }
    std::cout << "FINISHED!" << std::endl;

    std::cout << "Normalizing input... " << std::flush;
    normed_samples.reserve(samples.size());
    auto normalizer = data::make_normalizer();
    normalizer.train(samples);
    for (auto &r : samples) {
        normed_samples.push_back(normalizer(r));
    }
    std::cout << "FINISHED!" << std::endl;

    std::cout << "Training classifier... " << std::flush;
    classifier.set_number_of_centers(num_clusters);
    dlib::pick_initial_centers(num_clusters, initial_centers, normed_samples, classifier.get_kernel());
    classifier.train(normed_samples, initial_centers);
    std::cout << "FINISHED!" << std::endl;

    // Classify input data and save 1 file per class
    std::cout << "Saving classes... " << std::flush;
    std::vector<std::ofstream> classes;
    classes.reserve(num_clusters);
    for (unsigned i = 0; i < num_clusters; ++i) {
        classes.emplace_back(model_path + "." + std::to_string(i));
    }

    for (auto &s : samples) {
        auto predicted_cluster = classifier(normalizer(s));
        classes[predicted_cluster] << dlib::csv << dlib::trans(s);
    }
    std::cout << "FINISHED!" << std::endl;

    // Serialize models
    std::cout << "Saving models... " << std::flush;
    dlib::serialize(model_path + ".mod") << classifier;
    dlib::serialize(model_path + ".nrm") << normalizer;
    std::cout << "FINISHED!" << std::endl;
}