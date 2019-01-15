#ifndef SIM_H
#define SIM_H
class SimStateGPU{
    public:
        SimStateGPU(float *sigma, int * aux_data, int argc, char *argv[]);
        void step();
        float read_pressure();
        ~SimStateGPU();
        int GetWidth();
        int GetHeight();
        float GetPressure(int x, int y);
    private:
        int               deviceCount;
        int               targetDevice;

        float            *bufferP_in     ;
        float            *bufferVx_in    ;
        float            *bufferVy_in    ;

        float            *bufferP_out     ;
        float            *bufferVx_out    ;
        float            *bufferVy_out    ;

        float            *buffersSigma_in    ;

        int             *bufferAux_in    ;
        
        void gpu_step();

        void read_back();


        void init(float *sigma, int * aux_data, int argc, char *argv[]);

};

#endif