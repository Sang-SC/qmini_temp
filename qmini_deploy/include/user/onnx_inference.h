#ifndef ONNX_INFERENCE_H
#define ONNX_INFERENCE_H

#include <eigen3/Eigen/Dense>
#include "onnxruntime/onnxruntime_cxx_api.h"

class OnnxInference {
private:
    int64_t batch_size = 1;
    std::vector<const char *> input_node_names = {"input"};
    std::vector<const char *> output_node_names = {"output"};
    int64_t input_dim;
    int64_t output_dim;
    std::vector<int64_t> input_tensor_dims;
    size_t input_tensor_size;

public:
    void init(int obs_space, int act_space) {
        input_dim = obs_space;
        output_dim = act_space;
        input_tensor_dims = {batch_size, input_dim};
        input_tensor_size = batch_size * input_dim;
    }

    Eigen::Matrix<float, Eigen::Dynamic, 1> inference(Ort::Session *session, Eigen::Matrix<float, Eigen::Dynamic, 1> observation) {
        std::vector<float> input_tensor_values(input_tensor_size);
        for (unsigned int i = 0; i < input_tensor_size; i++)
            input_tensor_values[i] = observation[i];
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(),
                                                                  input_tensor_size, input_tensor_dims.data(), input_tensor_dims.size());
        std::vector<Ort::Value> ort_inputs;
        ort_inputs.push_back(std::move(input_tensor));
        auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_node_names.data(), ort_inputs.data(),
                                           ort_inputs.size(), output_node_names.data(), output_node_names.size());
        Eigen::Map<Eigen::Matrix<float, -1, 1> > net_out(output_tensors[0].GetTensorMutableData<float>(), output_dim, 1);
        return net_out;
    }
};

#endif