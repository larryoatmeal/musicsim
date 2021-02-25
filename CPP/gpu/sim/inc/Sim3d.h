#ifndef SIM3D_H
#define SIM3D_H

#include <vector>
#include "vector_types.h"

#include "constants.h"
class Sim3D{
    public:
        Sim3D(int dimx, int dimy, int dimz);
        void clean();
        void step(int n);
        void init();
        void reset();
        void restart();




        void setGPUNumber(int n);
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
        void setNumExcitorMultiple(float multipler);
        
        void setPressureMouth(float pressure);


        void setDT(float val);
        void setDS(float val);
        void setZN(float val);
        void setExcitorMode(int val);

        int getGlobalIndex(int x, int y, int z);

    private:
      float4 *m_bufferSrc;
      float4 *m_bufferDst;
      float4 *m_bufferIn;
      float4 *m_bufferOut;
      
      int *m_bufferAux;
      int *m_bufferInAux;

      float *m_audioBuffer;

      int m_i_global_listener;
      int m_i_global_p_bore;
      float m_p_mouth;


      int m_dimx;
      int m_dimy;
      int m_dimz;

      int m_i;
      int m_gpuNumber;

      int m_volumeSize;

      float m_multiplier = 50;

      dim3              m_dimBlock;
      dim3              m_dimGrid;

      int getStrideY();
      int getStrideZ();

      int* getAux();
      void calculateAndUpdateAux(int* aux);

      //
      std::vector< std::vector <int> > m_scheduledWalls;


    //   float m_dt;
    //   float m_ds;
    //   float m_zn;
        
};


#endif