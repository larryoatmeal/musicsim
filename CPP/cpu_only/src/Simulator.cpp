#include "Simulator.h"
#include "SimState.h"
#ifdef PYTHON

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

#endif

#ifdef PYTHON
namespace py = pybind11;

#endif

void Simulator::init(){
  #ifdef PYTHON
  py::print("Hello, World!"); 
  #endif
  sim.init_default_walls();
  #ifdef PYTHON
  py::print("Initialized default walls"); 
  #endif
  simState.init(sim.GetSigma(), sim.GetAuxData(), 0, NULL);
  #ifdef PYTHON
  py::print("Initialized GPU"); 
  #endif
}
std::vector<float> Simulator::run(int iter){
  #ifdef PYTHON
  py::print("Simulating..."); 
  #endif

  for(int i = 0; i < iter; i++){
    simState.step();
  }
  #ifdef PYTHON
  py::print("Reading back..."); 
  #endif

  std::vector<float> v = simState.read_back();
  #ifdef PYTHON
  py::print("Clearing..."); 
  #endif
  simState.clear();
  return v;
}

void Simulator::setWall(int x, int y){
  #ifdef PYTHON
  py::print("Setting wall"); 
  #endif
  sim.setWall(x, y, 1);
  simState.setAux(x, y, sim.GetAux(x, y));
}
void Simulator::clearWall(int x, int y){
  #ifdef PYTHON
  py::print("Clearing wall"); 
  #endif
  sim.setWall(x, y, 0);
  simState.setAux(x, y, sim.GetAux(x, y));
}



// #ifdef LOCAL
// void Simulator::init(){
//     simState.init_default_walls();
// }



// std::vector<float> Simulator::run(int iter){
//   std::vector<float> v(iter);
//   for(int i = 0; i < iter; i++){
//     simState.step();
//     v.push_back(simState.read_pressure());
//   }
//   simState.clear();
//   return v;
// }

// void Simulator::setWall(int x, int y){
//   #ifdef PYTHON
//     py::print("Hello, World!"); 
//   #endif

//   simState.setWall(x, y, 1);
// }
// void Simulator::clearWall(int x, int y){
//   simState.setWall(x, y, 0);
// }

// #else
// void Simulator::init(){
//   #ifdef PYTHON
//   py::print("Hello, World!"); 
//   #endif

//   SimState sim;
//   sim.init_default_walls();
//   #ifdef PYTHON
//   py::print("Initialized default walls"); 
//   #endif
//   simState.init(sim.GetSigma(), sim.GetAuxData(), 0, NULL);
//   #ifdef PYTHON
//   py::print("Initialized GPU"); 
//   #endif
// }
// std::vector<float> Simulator::run(int iter){
//   #ifdef PYTHON
//   py::print("Simulating..."); 
//   #endif

//   for(int i = 0; i < iter; i++){
//     simState.step();
//   }
//   #ifdef PYTHON
//   py::print("Reading back..."); 
//   #endif

//   std::vector<float> v = simState.read_back();
//   #ifdef PYTHON
//   py::print("Clearing..."); 
//   #endif
//   simState.clear();
//   return v;
// }

// void Simulator::setWall(int x, int y){

// }
// void Simulator::clearWall(int x, int y){

// }
// #endif

