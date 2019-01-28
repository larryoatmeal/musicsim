#include <cstring>
#include <iostream>

#include "Image.h"

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>      // std::stringstream
#include <iomanip>
#include <stdlib.h>

#include "SimState.h"
#include <cxxopts.hpp>
#include <cmath>
#include "AudioFile.h"


#ifndef LOCAL
#include "Sim.h"
#endif

int main(int argc, char **argv) {

    cxxopts::Options options("Simulation", "Instrument sim");
    options.add_options()
    ("h,help", "Print help")
    ("f,file", "File name", cxxopts::value<std::string>())
    ("d,debug", "Debug images")
    ("c,cpu", "cpu version")

    ("n,samples", "Number samples", cxxopts::value<int>(), "N")

    ;
    cxxopts::ParseResult result = options.parse(argc, argv);

    if (result.count("help"))
    {
      std::cout << options.help({"", "Group"}) << std::endl;
      exit(0);
    }


    bool debug = result.count("debug");
    if(result.count("debug")){
        std::cout << "DEBUG MODE: ON" << std::endl;
    }

    bool cpu = result.count("cpu");
    if(result.count("cpu")){
        std::cout << "CPU_ONLY: ON" << std::endl;
    }

    SimState sim;

    std::vector<float> audioBuffer;

    int N = 128000;

    if(!cpu){
        #ifdef LOCAL
          std::cout << "NO GPU CONNECTED" << std::endl;
        #else
          SimStateGPU simGPU(sim.GetSigma(), sim.GetAuxData(), argc, argv);
          for(int i = 0; i < N; i++){
              simGPU.step();
              if(i % (N/100) == 0){
                  std::cout << i * 100 /N << "%" << std::endl;
              }
          }
          audioBuffer = simGPU.read_back();
        #endif
        return 0;
    }
    else{
        if (result.count("samples"))
        {
            N = result["samples"].as<int>();
        }
        std::cout << "N = " << N << std::endl;

        int SKIP = 1000;
        std::vector<std::string> images;

        audioBuffer = std::vector<float>(N);
        for(int i = 0; i < N; i++){
            sim.step();

            audioBuffer[i] = sim.read_pressure();
            if(i % (N/100) == 0){
                std::cout << i * 100 /N << "%" << std::endl;
            }

            if(debug){
                if(i % SKIP == 0){
                    // std::cout << (float) i/N << std::endl;
                    std::stringstream formattedNumber;

                    formattedNumber << std::setfill('0') << std::setw(6) << i/SKIP;

                    std::string s = "out/img" + formattedNumber.str() + ".png";
                    Image image(sim.GetWidth(), sim.GetHeight());
                    for(int x = 0; x < sim.GetWidth(); x++){
                        for(int y = 0; y < sim.GetHeight(); y++){
                            float p = sim.GetPressure(x, y);
                            float col = std::max(std::min(p * 1000, 1.0f), -1.0f);
                            if(col < 0){
                                Vector3f color(0, -col, 0);
                                image.setPixel(x, y, color);
                            }else{
                                Vector3f color(col, 0, 0);
                                image.setPixel(x, y, color);
                            }
                        }
                    }
                    images.push_back(s);

                    image.savePNG(s);
                }
            }
        }

        if(debug){
            std::stringstream video_cmd;
            int framerate = 24;
            video_cmd << "ffmpeg" << " " << "-framerate" << " " << 24 << " -i out/img%06d.png -pix_fmt yuv420p out/output.mp4";
            std::string video = video_cmd.str();
            std::cout << video << std::endl;

            system(video_cmd.str().c_str());

            for(int i = 0; i < images.size(); i++){
                std::string s = images[i];
                std::string rm = "rm " + s;
                system(rm.c_str());
            }
        }

    }
    //normalized audio
    float max_amp = 0;
    for(int i = 0; i < audioBuffer.size(); i++){
        float amp = std::abs(audioBuffer[i]);

        max_amp = std::max(amp, max_amp);
    }
    for(int i = 0; i < audioBuffer.size(); i++){
        audioBuffer[i] /= max_amp;
    }


    int SUBSAMPLED_N = N/ SAMPLE_EVERY_N;

    AudioFile<float> audioFile;
    AudioFile<float>::AudioBuffer buffer;
    buffer.resize (1);
    buffer[0].resize (SUBSAMPLED_N);
    for(int i = 0; i < SUBSAMPLED_N; i++){
        buffer[0][i] = audioBuffer[i * SAMPLE_EVERY_N];
    }
    audioFile.setAudioBuffer(buffer);
    audioFile.printSummary();

    audioFile.save ("out/audio.wav");
  return 0;
}
