#include "Sim.h"

#include "iostream"

int main(int argc, char *argv[]){
  std::cout << "HELLO" << std::endl;

  SimState s(0, 0, argc, argv);
  return 0;
}