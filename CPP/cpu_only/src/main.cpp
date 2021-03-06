#include <cstring>
#include <iostream>

#include "Image.h"

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>      // std::stringstream
#include <iomanip>
#include <stdlib.h>

// #include "SimState.h"
#include <cxxopts.hpp>
#include <cmath>
#include "AudioFile.h"
// #include "Simulator.h"


#include "Sim3d.h"
#include "Sim3dCPU.h"



#ifdef PYTHON
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#endif


#ifndef LOCAL
// #include "Sim.h"
#endif



#ifdef PYTHON

namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}

int test(){
    // Simulator sim;
}

PYBIND11_MODULE(pytest, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function which adds two numbers");

    // py::class_<Simulator>(m, "Simulator")
    //     .def(py::init<>())
    //     .def("init", &Simulator::init)
    //     .def("run", &Simulator::run)
    //     .def("setWall", &Simulator::setWall)
    //     .def("clearWall", &Simulator::clearWall);

    py::class_<Sim3D>(m, "sim3d")
        .def(py::init<const int &, const int &, const int &>())
        .def("init", &Sim3D::init)
        .def("clean", &Sim3D::clean)
        .def("reset", &Sim3D::reset)
        .def("restart", &Sim3D::restart)
        .def("setSigma", &Sim3D::setSigma)
        .def("setWall", &Sim3D::setWall)
        .def("readBackAux", &Sim3D::readBackAux)
        .def("readBackAudio", &Sim3D::readBackAudio)
        .def("readBackData", &Sim3D::readBackData)
        .def("scheduleWall", &Sim3D::scheduleWall)
        .def("writeWalls", &Sim3D::writeWalls)
        .def("setPBore", &Sim3D::setPBore)
        .def("setPressureMouth", &Sim3D::setPressureMouth)
        .def("setExcitor", &Sim3D::setExcitor)
        .def("setExcitors", &Sim3D::setExcitors)
        .def("step", &Sim3D::step)
        .def("setListener", &Sim3D::setListener)
        .def("DT", &Sim3D::setDT)
        .def("DS", &Sim3D::setDS)
        .def("ZN", &Sim3D::setZN)
        .def("setExcitorMode", &Sim3D::setExcitorMode)
        .def("readBackDataCoords", &Sim3D::readBackDataCoords)
        .def("setGPUNumber", &Sim3D::setGPUNumber)
        .def("setNumExcitorMultiple", &Sim3D::setNumExcitorMultiple);

    py::class_<Sim3DCPU>(m, "sim3dCPU")
        .def(py::init<const int &, const int &, const int &>())
        .def("init", &Sim3DCPU::init)
        .def("clean", &Sim3DCPU::clean)
        .def("reset", &Sim3DCPU::reset)
        .def("restart", &Sim3DCPU::restart)
        .def("setSigma", &Sim3DCPU::setSigma)
        .def("setWall", &Sim3DCPU::setWall)
        .def("readBackAux", &Sim3DCPU::readBackAux)
        .def("readBackAudio", &Sim3DCPU::readBackAudio)
        .def("readBackData", &Sim3DCPU::readBackData)
        .def("scheduleWall", &Sim3DCPU::scheduleWall)
        .def("writeWalls", &Sim3DCPU::writeWalls)
        .def("setPBore", &Sim3DCPU::setPBore)
        .def("setPressureMouth", &Sim3DCPU::setPressureMouth)
        .def("setExcitor", &Sim3DCPU::setExcitor)
        .def("setExcitors", &Sim3DCPU::setExcitors)
        .def("step", &Sim3DCPU::step)
        .def("setListener", &Sim3DCPU::setListener)
        .def("DT", &Sim3DCPU::setDT)
        .def("DS", &Sim3DCPU::setDS)
        .def("ZN", &Sim3DCPU::setZN)
        // .def("setGPUNumber", &Sim3DCPU::setGPUNumber)
        .def("readBackDataCoords", &Sim3DCPU::readBackDataCoords);

        


}
#endif






void debug_save(std::string name, float* data, int w, int h, int i){
    // std::ofstream myfile;
    // std::stringstream file_name;
    // file_name << "out/cpp_" << name << "_" << i << ".csv";

    // myfile.open (file_name.str());
    // for(int y = 0; y < h; y++){
    //     for(int x = 0; x < w; x++){
    //         myfile << data[index_of_padded(x, y)] << ",";
    //     }
    //     myfile << "\n";
    // }
    // myfile.close();
}


int main(int argc, char **argv) {
    Sim3D sim3d(128, 128, 128);

    std::cout << "INIT START" << std::endl;
    
    sim3d.init();

    // // for(int x = 0; x < 32; x++){
    // //     for(int y = 0; y < 32; y++){
    // //         for(int z = 0; z < 32; z++){
    // //             sim3d.setWall(x, y, z, 1);
    // //         }
    // //     }
    // // }

    // std::vector<int> aux = sim3d.readBackAux();

    std::cout << "INIT END" << std::endl;

    // // sim3d.setWall(10, 10, 10, 1);

    // sim3d.clean();
    

    // cxxopts::Options options("Simulation", "Instrument sim");
    // options.add_options()
    // ("h,help", "Print help")
    // ("f,file", "File name", cxxopts::value<std::string>())
    // ("d,debug", "Debug images")
    // ("c,cpu", "cpu version")
    // ("i,inspect", "full inspect")

    // ("n,samples", "Number samples", cxxopts::value<int>(), "N")

    // ;
    // cxxopts::ParseResult result = options.parse(argc, argv);

    // if (result.count("help"))
    // {
    //   std::cout << options.help({"", "Group"}) << std::endl;
    //   exit(0);
    // }


    // bool debug = result.count("debug");
    // if(result.count("debug")){
    //     std::cout << "DEBUG MODE: ON" << std::endl;
    // }

    // bool cpu = result.count("cpu");
    // if(result.count("cpu")){
    //     std::cout << "CPU_ONLY: ON" << std::endl;
    // }

    // bool csv_out = result.count("inspect");
    // if(csv_out){
    //     std::cout << "CSV_OUT: ON" << std::endl;
    // }

    // SimState sim;
    // sim.init_default_walls();

    // std::vector<float> audioBuffer;

    // int N = 128000;
    // if (result.count("samples"))
    // {
    //     N = result["samples"].as<int>();
    // }
    // std::cout << "N = " << N << std::endl;
    

    // if(!cpu){
    //     #ifdef LOCAL
    //       std::cout << "NO GPU CONNECTED" << std::endl;
    //       return 0;
    //     #else
    //       SimStateGPU simGPU;
    //       simGPU.init(sim.GetSigma(), sim.GetAuxData(), 0, NULL);
    //       for(int i = 0; i < N; i++){
    //           simGPU.step();
    //           if(i % (N/100) == 0){
    //               std::cout << i * 100 /N << "%" << std::endl;
    //           }
    //       }
    //       audioBuffer = simGPU.read_back();
    //     #endif
    // }
    // else{
        

    //     int SKIP = 1000;
    //     std::vector<std::string> images;

    //     audioBuffer = std::vector<float>(N);
    //     for(int i = 0; i < N; i++){
    //         sim.step();

    //         audioBuffer[i] = sim.read_pressure();
    //         if(i % (N/100) == 0){
    //             std::cout << i * 100 /N << "%" << std::endl;
    //         }

    //         if(csv_out){
    //             // std::cout << "CSV_OUT" << std::endl;
    //             debug_save("p", sim.p_prev, sim.GetWidth(), sim.GetHeight(), i);
    //             debug_save("v_x", sim.v_x_prev, sim.GetWidth(), sim.GetHeight(), i);
    //             debug_save("v_y", sim.v_y_prev, sim.GetWidth(), sim.GetHeight(), i);
    //             debug_save("beta", sim.beta, sim.GetWidth(), sim.GetHeight(), i);

    //         }

    //         if(debug){
    //             if(i % SKIP == 0){
    //                 // std::cout << (float) i/N << std::endl;
    //                 std::stringstream formattedNumber;

    //                 formattedNumber << std::setfill('0') << std::setw(6) << i/SKIP;

    //                 std::string s = "out/img" + formattedNumber.str() + ".png";
    //                 Image image(sim.GetWidth(), sim.GetHeight());
    //                 for(int x = 0; x < sim.GetWidth(); x++){
    //                     for(int y = 0; y < sim.GetHeight(); y++){
    //                         float p = sim.GetPressure(x, y);
    //                         float col = std::max(std::min(p/1000, 1.0f), -1.0f);
    //                         if(col < 0){
    //                             Vector3f color(0, -col, 0);
    //                             image.setPixel(x, y, color);
    //                         }else{
    //                             Vector3f color(col, 0, 0);
    //                             image.setPixel(x, y, color);
    //                         }
    //                     }
    //                 }
    //                 images.push_back(s);

    //                 image.savePNG(s);
    //             }
    //         }
    //     }

    //     if(debug){
    //         std::stringstream video_cmd;
    //         int framerate = 24;
    //         video_cmd << "ffmpeg" << " " << "-framerate" << " " << 24 << " -i out/img%06d.png -pix_fmt yuv420p out/output.mp4";
    //         std::string video = video_cmd.str();
    //         std::cout << video << std::endl;

    //         system(video_cmd.str().c_str());

    //         for(int i = 0; i < images.size(); i++){
    //             std::string s = images[i];
    //             std::string rm = "rm " + s;
    //             system(rm.c_str());
    //         }
    //     }

    // }
    // //normalized audio
    // float max_amp = 0;
    // for(int i = 0; i < audioBuffer.size(); i++){
    //     float amp = std::abs(audioBuffer[i]);

    //     max_amp = std::max(amp, max_amp);
    // }
    // for(int i = 0; i < audioBuffer.size(); i++){
    //     audioBuffer[i] /= max_amp;
    // }

    // std::cout << "FINISHED" << std::endl;


    // int SUBSAMPLED_N = N/ SAMPLE_EVERY_N;

    // AudioFile<float> audioFile;
    // AudioFile<float>::AudioBuffer buffer;
    // buffer.resize (1);
    // buffer[0].resize (SUBSAMPLED_N);
    // for(int i = 0; i < SUBSAMPLED_N; i++){
    //     buffer[0][i] = audioBuffer[i * SAMPLE_EVERY_N];
    // }
    // audioFile.setAudioBuffer(buffer);
    // audioFile.printSummary();

    // audioFile.save ("out/audio.wav");
  return 0;
}
