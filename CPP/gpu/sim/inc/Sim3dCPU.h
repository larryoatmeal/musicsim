#ifndef SIM3DCPU_H
#define SIM3DCPU_H

#include <vector>
#include "vector_types.h"

#include "constants.h"

struct Velocity{
  float x;
  float y;
  float z;
  float& operator[] (int index){
    if(index == 0){
      return x;
    }
    else if(index == 1){
      return y;
    }
    else if(index == 2){
      return z;
    }
    else{
      exit(0);
    }
  };
};


struct State{
  Velocity velocity;
  float pressure;
};

struct Coord{
  int x;
  int y;
  int z;
};


struct Cell{
  State oldState;
  State newState;
  bool isWall;
  bool isExcitor;
  float sigma;
  int beta(){
    if(isWall || isExcitor){
      return 0;
    }else{
      return 1;
    }
  }
};

class Sim3DCPU{
    public:
        Sim3DCPU(int dimx, int dimy, int dimz);
        void clean();
        void step(int n);
        void init();
        void reset();
        void restart();



        std::vector<float> readBackAudio();
        std::vector< std::vector<float> > readBackData();
        std::vector< std::vector<float> > readBackDataCoords(std::vector< std::vector<int> > coords);

        std::vector<int> readBackAux();

        void scheduleWall(int x, int y, int z, int val);
        void writeWalls();

        void setSigma(int x, int y, int z, int level);
        void setWall(int x, int y, int z, int val);

        void setListener(int x, int y, int z);
        void setExcitor(int x, int y, int z, int val);

        void setExcitors(std::vector< std::vector<int> > excitors);

        void setPBore(int x, int y, int z);

        
        void setPressureMouth(float pressure);


        void setDT(float val);
        void setDS(float val);
        void setZN(float val);
        int getGlobalIndex(int x, int y, int z);

    private:
      std::vector<Cell> m_cells;

      std::vector<float> m_audioBuffer;

      
      float m_p_mouth;

      float m_dt;
      float m_ds;
      float m_zn;
      int m_numExcitors;

      int m_dimx;
      int m_dimy;
      int m_dimz;

      int m_i;

      int m_volumeSize;

      Coord listen;
      Coord pBore;

      dim3              m_dimBlock;
      dim3              m_dimGrid;

      int getStrideY();
      int getStrideZ();

      std::vector< std::vector <int> > m_scheduledWalls;


    Cell DAT(int x, int y, int z);

    float P(int x, int y, int z);

    Velocity V(int x, int y, int z);
    float SIG(int x, int y, int z);

    int BETA(int x, int y, int z);
    float SPRIME(int x, int y, int z);
    float vStep(int x, int y, int z, Velocity vb, int axis, float gradCoeff);
};


#endif