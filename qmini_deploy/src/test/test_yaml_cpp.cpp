#include <yaml-cpp/yaml.h>
#include <filesystem> 
#include <vector>
#include <string>
#include <iostream>

using std::cout , std::endl;
using std::string;
using std::vector;
namespace fs = std::filesystem;

int main() {
    string config_path = fs::path(__FILE__).parent_path().string() + "/../../config/test_yaml_cpp.yaml";
    YAML::Node config = YAML::LoadFile(config_path);

    string name = config["name"].as<string>();
    cout << "Name: " << name << endl;
    
    int age = config["age"].as<int>();
    cout << "Age: " << age << endl;
    
    bool sim_flag = config["sim_flag"].as<bool>();
    cout << "Sim Flag: " << (sim_flag ? "true" : "false") << endl;
    
    vector<float> kp = config["kp"].as<vector<float>>();
    cout << "Kp: [";
    for (size_t i = 0; i < kp.size(); ++i) {
        cout << kp[i];
        if (i < kp.size() - 1) cout << ", ";
    }
    cout << "]" << endl;
    
    vector<float> kd = config["kd"].as<vector<float>>();
    cout << "Kd: [";
    for (size_t i = 0; i < kd.size(); ++i) {
        cout << kd[i];
        if (i < kd.size() - 1) cout << ", ";
    }
    cout << "]" << endl;
    return 0;
}