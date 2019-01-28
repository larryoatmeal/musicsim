#ifndef SIM_H
#define SIM_H

#include <vector>


class SimStateGPU{
    public:
        SimStateGPU(float *sigma, int * aux_data, int argc, char *argv[]);
        void step();
        float read_pressure();
        ~SimStateGPU();
        int GetWidth();
        int GetHeight();
        float GetPressure(int x, int y);
        std::vector<float> read_back();

        float            *bufferP_in     ;
        float            *bufferVx_in    ;
        float            *bufferVy_in    ;

        float            *bufferP_out     ;
        float            *bufferVx_out    ;
        float            *bufferVy_out    ;

        float            *buffersSigma_in    ;

        int             *bufferAux_in    ;
        float           *bufferAudio;
    private:
        int               deviceCount;
        int               targetDevice;

        
        int             iter;
        
        void gpu_step();


        void init(float *sigma, int * aux_data, int argc, char *argv[]);

};

#endif