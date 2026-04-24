"""ONNX to C++ inference code generator.

Generates standalone C++ code that:
  - Loads the ONNX model via ONNX Runtime C API
  - Allocates input/output tensors with correct shapes and types
  - Runs inference
  - Prints output summary

For edge deployment where Python is not available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("isat.codegen")


ONNX_TYPE_TO_CPP = {
    1: ("float", "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT"),
    2: ("uint8_t", "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8"),
    3: ("int8_t", "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8"),
    6: ("int32_t", "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32"),
    7: ("int64_t", "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64"),
    10: ("Ort::Float16_t", "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16"),
    11: ("double", "ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE"),
    9: ("bool", "ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL"),
}


@dataclass
class CodegenResult:
    model_path: str
    output_path: str
    cmake_path: str
    lines_of_code: int
    inputs: list[dict] = field(default_factory=list)
    outputs: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"  Model       : {self.model_path}",
            f"  C++ file    : {self.output_path}",
            f"  CMake file  : {self.cmake_path}",
            f"  Lines       : {self.lines_of_code}",
            f"  Inputs      : {len(self.inputs)}",
            f"  Outputs     : {len(self.outputs)}",
            f"",
            f"  Build instructions:",
            f"    mkdir build && cd build",
            f"    cmake .. -DONNXRUNTIME_DIR=/path/to/onnxruntime",
            f"    make",
            f"    ./isat_inference ../model.onnx",
        ]
        return "\n".join(lines)


class CppCodeGenerator:
    """Generate C++ inference code from ONNX model."""

    def __init__(self, model_path: str):
        import onnx
        self.model_path = model_path
        self.model = onnx.load(str(model_path), load_external_data=False)

    def generate(self, output_dir: str = "") -> CodegenResult:
        out_dir = Path(output_dir or ".")
        out_dir.mkdir(parents=True, exist_ok=True)

        graph = self.model.graph
        inputs_info = self._parse_io(graph.input, graph.initializer)
        outputs_info = self._parse_io(graph.output, [])

        cpp_code = self._gen_cpp(inputs_info, outputs_info)
        cmake_code = self._gen_cmake()

        cpp_path = out_dir / "isat_inference.cpp"
        cmake_path = out_dir / "CMakeLists.txt"
        cpp_path.write_text(cpp_code)
        cmake_path.write_text(cmake_code)

        return CodegenResult(
            model_path=self.model_path,
            output_path=str(cpp_path),
            cmake_path=str(cmake_path),
            lines_of_code=cpp_code.count("\n"),
            inputs=inputs_info, outputs=outputs_info,
        )

    def _parse_io(self, io_list, initializers) -> list[dict]:
        init_names = {i.name for i in initializers}
        result = []
        for item in io_list:
            if item.name in init_names:
                continue
            tt = item.type.tensor_type
            elem_type = tt.elem_type
            cpp_type, ort_type = ONNX_TYPE_TO_CPP.get(elem_type, ("float", "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT"))
            shape = []
            for dim in tt.shape.dim:
                if dim.dim_value > 0:
                    shape.append(dim.dim_value)
                else:
                    shape.append(1)
            result.append({
                "name": item.name, "cpp_type": cpp_type,
                "ort_type": ort_type, "shape": shape,
                "elem_type": elem_type,
            })
        return result

    def _gen_cpp(self, inputs: list[dict], outputs: list[dict]) -> str:
        lines = []
        lines.append('#include <onnxruntime_cxx_api.h>')
        lines.append('#include <iostream>')
        lines.append('#include <vector>')
        lines.append('#include <chrono>')
        lines.append('#include <numeric>')
        lines.append('#include <algorithm>')
        lines.append('#include <cmath>')
        lines.append('#include <cstdlib>')
        lines.append('')
        lines.append('int main(int argc, char* argv[]) {')
        lines.append('    if (argc < 2) {')
        lines.append('        std::cerr << "Usage: " << argv[0] << " <model.onnx> [runs]" << std::endl;')
        lines.append('        return 1;')
        lines.append('    }')
        lines.append('    const char* model_path = argv[1];')
        lines.append('    int runs = argc > 2 ? std::atoi(argv[2]) : 10;')
        lines.append('')
        lines.append('    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "isat");')
        lines.append('    Ort::SessionOptions opts;')
        lines.append('    opts.SetIntraOpNumThreads(0);')
        lines.append('    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);')
        lines.append('')
        lines.append('    Ort::Session session(env, model_path, opts);')
        lines.append('    Ort::AllocatorWithDefaultOptions allocator;')
        lines.append('')

        for i, inp in enumerate(inputs):
            size = 1
            for d in inp["shape"]:
                size *= d
            shape_str = "{" + ", ".join(str(d) for d in inp["shape"]) + "}"
            lines.append(f'    std::vector<int64_t> input{i}_shape = {shape_str};')
            lines.append(f'    std::vector<{inp["cpp_type"]}> input{i}_data({size}, 1);')
            lines.append(f'    auto input{i}_tensor = Ort::Value::CreateTensor<{inp["cpp_type"]}>(')
            lines.append(f'        allocator.GetInfo(), input{i}_data.data(), {size},')
            lines.append(f'        input{i}_shape.data(), {len(inp["shape"])});')
            lines.append('')

        input_names_arr = ", ".join(f'"{inp["name"]}"' for inp in inputs)
        output_names_arr = ", ".join(f'"{out["name"]}"' for out in outputs)
        lines.append(f'    const char* input_names[] = {{{input_names_arr}}};')
        lines.append(f'    const char* output_names[] = {{{output_names_arr}}};')
        lines.append('')

        lines.append(f'    std::vector<Ort::Value> input_tensors;')
        for i in range(len(inputs)):
            lines.append(f'    input_tensors.push_back(std::move(input{i}_tensor));')
        lines.append('')

        lines.append('    // Warmup')
        lines.append('    for (int w = 0; w < 3; w++) {')
        lines.append(f'        session.Run(Ort::RunOptions{{nullptr}}, input_names,')
        lines.append(f'                    input_tensors.data(), {len(inputs)},')
        lines.append(f'                    output_names, {len(outputs)});')
        lines.append('    }')
        lines.append('')

        lines.append('    std::vector<double> latencies;')
        lines.append('    for (int r = 0; r < runs; r++) {')
        lines.append('        auto t0 = std::chrono::high_resolution_clock::now();')
        lines.append(f'        auto outputs = session.Run(Ort::RunOptions{{nullptr}},')
        lines.append(f'                    input_names, input_tensors.data(), {len(inputs)},')
        lines.append(f'                    output_names, {len(outputs)});')
        lines.append('        auto t1 = std::chrono::high_resolution_clock::now();')
        lines.append('        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();')
        lines.append('        latencies.push_back(ms);')
        lines.append('    }')
        lines.append('')

        lines.append('    double mean = std::accumulate(latencies.begin(), latencies.end(), 0.0) / runs;')
        lines.append('    std::sort(latencies.begin(), latencies.end());')
        lines.append('    double p50 = latencies[runs / 2];')
        lines.append('    double p95 = latencies[(int)(runs * 0.95)];')
        lines.append('    double p99 = latencies[(int)(runs * 0.99)];')
        lines.append('')
        lines.append('    std::cout << "=== ISAT C++ Inference Benchmark ===" << std::endl;')
        lines.append('    std::cout << "Model : " << model_path << std::endl;')
        lines.append('    std::cout << "Runs  : " << runs << std::endl;')
        lines.append('    std::cout << "Mean  : " << mean << " ms" << std::endl;')
        lines.append('    std::cout << "P50   : " << p50 << " ms" << std::endl;')
        lines.append('    std::cout << "P95   : " << p95 << " ms" << std::endl;')
        lines.append('    std::cout << "P99   : " << p99 << " ms" << std::endl;')
        lines.append('    return 0;')
        lines.append('}')
        return "\n".join(lines)

    def _gen_cmake(self) -> str:
        return """cmake_minimum_required(VERSION 3.14)
project(isat_inference LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(onnxruntime QUIET)
if(NOT onnxruntime_FOUND)
    if(DEFINED ENV{ONNXRUNTIME_DIR})
        set(ONNXRUNTIME_DIR $ENV{ONNXRUNTIME_DIR})
    elseif(NOT DEFINED ONNXRUNTIME_DIR)
        message(FATAL_ERROR "Set -DONNXRUNTIME_DIR=/path/to/onnxruntime or ONNXRUNTIME_DIR env")
    endif()
    include_directories(${ONNXRUNTIME_DIR}/include)
    link_directories(${ONNXRUNTIME_DIR}/lib)
endif()

add_executable(isat_inference isat_inference.cpp)
target_link_libraries(isat_inference onnxruntime)
"""
