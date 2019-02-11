#include "Simulator.h"
#include "SimState.h"
#ifdef LOCAL
void Simulator::init(){
  simState = SimState();
}

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
  simState.setWall(x, y, 1);
}
void Simulator::clearWall(int x, int y){
  simState.setWall(x, y, 0);
}

#else
void Simulator::init(){

}
std::vector<float> Simulator::run(int iter){

}

void Simulator::setWall(int x, int y){

}
#endif

