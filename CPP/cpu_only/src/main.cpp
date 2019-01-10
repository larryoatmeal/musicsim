#include <cstring>
#include <iostream>

#include "Image.h"

#include <iostream>
#include <string>
#include <math.h>
#include <fstream>
#include <sstream>      // std::stringstream
#include <iomanip>
#include <stdlib.h>

#include "SimState.h"

int main() {
  std::cout << "Hello world!" << std::endl;
  
  SimState sim;

  int N = 200000;
  int SKIP = 10;

  std::vector<std::string> images;

  for(int i = 0; i < N; i++){
    sim.step();
    if(i % SKIP == 0){
        std::cout << (float) i/N << std::endl;
        std::stringstream formattedNumber;

        formattedNumber << std::setfill('0') << std::setw(6) << i/SKIP;

        std::string s = "img" + formattedNumber.str() + ".png";
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


       
        // std::string video_cmd = "ffmpeg -framerate 24 -i img%03d.png output.mp4";


    }

    

  }

    std::stringstream video_cmd;
    int framerate = 24;
    video_cmd << "ffmpeg" << " " << "-framerate" << " " << 24 << " -i img%06d.png -pix_fmt yuv420p output.mp4";
    std::string video = video_cmd.str();
    std::cout << video << std::endl;

    system(video_cmd.str().c_str());

    for(int i = 0; i < images.size(); i++){
        std::string s = images[i];
        std::string rm = "rm " + s;
        system(rm.c_str());
    }

  return 0;
}
