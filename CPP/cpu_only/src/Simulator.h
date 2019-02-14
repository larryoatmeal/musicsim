#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <vector>

// #ifdef LOCAL
// #include "SimState.h"
// #else
#include "Sim.h"

// #endif

class Simulator{

public:



  void init();
  std::vector<float> run(int iter);
  void setWall(int x, int y);
  void clearWall(int x, int y);


private:
  // #ifdef LOCAL
  //   SimState simState;
  // #else
    SimStateGPU simState;
  // #endif

};

#endif