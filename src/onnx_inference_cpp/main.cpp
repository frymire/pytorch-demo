// Runs the exported FashionMNIST model from C++.
//
// The program reads model.onnx, reads one image from the FashionMNIST test
// files, and prints the class the model predicts. It uses ONNX Runtime and
// zlib. It needs no PyTorch and no Python.
//
// CMake downloads the test data, so the program needs no other setup.
// The data files stay gzip compressed, and zlib reads them in place.
//
// Usage: onnx_demo [model.onnx] [data directory] [image index]

#include <onnxruntime_cxx_api.h>

#include <zlib.h>

#include <array>
using std::array;

#include <cstdint>

#include <iostream>
using std::cout;

#include <stdexcept>
using std::runtime_error;

#include <string>
using std::string;

#include <vector>
using std::vector;

namespace {

// The ten FashionMNIST classes, in the order the model scores them.
constexpr array<const char*, 10> CLASSES = {
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
};

constexpr int IMAGE_WIDTH = 28;
constexpr int IMAGE_HEIGHT = 28;
constexpr int PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT;

// Close a gzip file even when an error throws.
class GzipFile {
public:
    explicit GzipFile(const string& path) : m_file(gzopen(path.c_str(), "rb")) {
        if (m_file == nullptr) throw runtime_error("Cannot open " + path);
    }
    ~GzipFile() { if (m_file != nullptr) gzclose(m_file); }
    GzipFile(const GzipFile&) = delete;
    GzipFile& operator=(const GzipFile&) = delete;
    [[nodiscard]] gzFile get() const { return m_file; }

private:
    gzFile m_file;
};

// ReSharper disable once CppParameterMayBeConst
void read_exactly(gzFile file, void* target, const int count) {
    if (gzread(file, target, static_cast<unsigned>(count)) != count) {
        throw runtime_error("The data file ended too early.");
    }
}

// The IDX file format stores its header numbers most significant byte first.
// Intel processors store them the other way round, so swap the order.
// ReSharper disable once CppParameterMayBeConst
uint32_t read_big_endian(gzFile file) {
    unsigned char bytes[4];
    read_exactly(file, bytes, 4);
    return
      (static_cast<uint32_t>(bytes[0]) << 24) |
      (static_cast<uint32_t>(bytes[1]) << 16) |
      (static_cast<uint32_t>(bytes[2]) << 8) |
      static_cast<uint32_t>(bytes[3]);
}

// Read one 28x28 image and scale each pixel from 0-255 into 0.0-1.0. This
// matches what v2.ToDtype(torch.float32, scale=True) does in Python.
vector<float> read_image(const string& path, const int index) {
    
    // ReSharper disable once CppLocalVariableMayBeConst
    GzipFile file(path);

    read_big_endian(file.get());  // magic number
    const uint32_t count = read_big_endian(file.get());
    read_big_endian(file.get());  // rows
    read_big_endian(file.get());  // columns
    if (index < 0 || static_cast<uint32_t>(index) >= count) { throw runtime_error("The image index is out of range."); }

    gzseek(file.get(), static_cast<off_t>(index) * PIXELS, SEEK_CUR);
    vector<unsigned char> raw(PIXELS);
    read_exactly(file.get(), raw.data(), PIXELS);

    vector<float> pixels(PIXELS);
    for (int i = 0; i < PIXELS; ++i) pixels[i] = static_cast<float>(raw[i]) / 255.0f;
    return pixels;
}

int read_label(const string& path, int index) {
    // ReSharper disable once CppLocalVariableMayBeConst
    GzipFile file(path);
    read_big_endian(file.get());                        // magic number
    read_big_endian(file.get());                        // count
    gzseek(file.get(), index, SEEK_CUR);
    unsigned char label = 0;
    read_exactly(file.get(), &label, 1);
    return label;
}

// Draw the image with text, so you can check the answer by eye.
void print_image(const vector<float>& pixels) {
    constexpr auto shades = " .:-=+*#%@";
    for (int row = 0; row < IMAGE_HEIGHT; ++row) {
        for (int col = 0; col < IMAGE_WIDTH; ++col) {
            const float value = pixels[row * IMAGE_WIDTH + col];
            cout << shades[static_cast<int>(value * 9.0f)];
        }
        cout << '\n';
    }
}

std::wstring widen(const string& text) { return {text.begin(), text.end()}; }

}  // namespace

int main(const int argc, char** argv) {

    try {

        const string model_path = argc > 1 ? argv[1] : DEFAULT_MODEL_PATH;
        const string data_dir = argc > 2 ? argv[2] : DATA_DIRECTORY;
        const int index = argc > 3 ? std::stoi(argv[3]) : 0;

        const vector<float> pixels = read_image(data_dir + "/t10k-images-idx3-ubyte.gz", index);
        const int actual = read_label(data_dir + "/t10k-labels-idx1-ubyte.gz", index);

        print_image(pixels);

        // Start ONNX Runtime and load the model.
        const Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_demo");
        const Ort::SessionOptions options;
        Ort::Session session(env, widen(model_path).c_str(), options);

        // Read the input and output names out of the file itself.
        const Ort::AllocatorWithDefaultOptions allocator;
        const auto input_name = session.GetInputNameAllocated(0, allocator);
        const auto output_name = session.GetOutputNameAllocated(0, allocator);
        const char* input_names[] = {input_name.get()};
        const char* output_names[] = {output_name.get()};

        // Shape the input as [batch, channel, height, width], the same shape
        // the Python model expects.
        constexpr array<int64_t, 4> shape = {1, 1, IMAGE_HEIGHT, IMAGE_WIDTH};
        const Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const Ort::Value input = Ort::Value::CreateTensor<float>(
            memory,
            const_cast<float*>(pixels.data()),
            pixels.size(),
            shape.data(),
            shape.size()
        );

        const auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 1);

        // The model returns ten raw scores, called logits. The largest one
        // marks the class the model picks.
        const auto* logits = outputs[0].GetTensorData<float>();
        int best = 0;
        for (int i = 1; i < 10; ++i) {
            if (logits[i] > logits[best]) { best = i; }
        }

        cout << "\nModel: " << model_path << '\n';
        cout << "Input name: " << input_names[0] << '\n';
        cout << "Output name: " << output_names[0] << "\n\n";
        cout << "Predicted: \"" << CLASSES[best] << "\"\nActual: \"" << CLASSES[actual] << "\"\n";
        return best == actual ? 0 : 1;

    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 2;
    }
}
