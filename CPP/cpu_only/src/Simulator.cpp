#include "Simulator.h"
#include "SimState.h"
#ifdef PYTHON

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

#endif


#ifdef LOCAL
void Simulator::init(){
}

#ifdef PYTHON
namespace py = pybind11;

#endif

std::vector<float> Simulator::run(int iter){
  std::vector<float> v(iter);
  for(int i = 0; i < iter; i++){
    simState.step();
    v.push_back(simState.read_pressure());
  }
  simState.clear();
  return v;
}

void Simulator::setWall(int x, int y){
  #ifdef PYTHON
    py::print("Hello, World!"); 
  #endif

  simState.setWall(x, y, 1);
}
void Simulator::clearWall(int x, int y){
  simState.setWall(x, y, 0);
}

#else
void Simulator::init(){
  SimState sim;
  simState = SimStateGPU(sim.GetSigma(), sim.GetAuxData(), 0, NULL);
}
std::vector<float> Simulator::run(int iter){
  for(int i = 0; i < iter; i++){
    simState.step();
  }

  std::vector<float> v = simState.read_back();

  simState.clear();
  return v;
}

void Simulator::setWall(int x, int y){

}
void Simulator::clearWall(int x, int y){

}
#endif

