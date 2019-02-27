#ifndef SIM3D_H
#define SIM3D_H

#include <vector>
#include "vector_types.h"

#include "constants.h"
class Sim3D{
    public:
        Sim3D(int dimx, int dimy, int dimz);
        void clean();
        void step();
        void init();
        void reset();

        std::vector<float> readBackAudio();
        std::vector< std::vector<float> > readBackData();
        std::vector<int> readBackAux();


        void setSigma(int x, int y, int z, int level);
        void setWall(int x, int y, int z, int val);
        int getGlobalIndex(int x, int y, int z);

    private:
      float4 *m_bufferSrc;
      float4 *m_bufferDst;
      float4 *m_bufferIn;
      float4 *m_bufferOut;
      
      int *m_bufferAux;
      int *m_bufferInAux;

      float *m_audioBuffer;


      int m_dimx;
      int m_dimy;
      int m_dimz;

      int m_i;

      int m_volumeSize;

      dim3              m_dimBlock;
      dim3              m_dimGrid;

      int getStrideY();
      int getStrideZ();
        
};


#endif