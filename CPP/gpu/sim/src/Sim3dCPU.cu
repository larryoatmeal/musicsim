#include "Sim3dCPU.h"
#include "constants.h"
#include <stdio.h>

#include <algorithm> 
const int MAX_AUDIO_SIZE = 44100 * 10;

Sim3DCPU::Sim3DCPU(int dimx, int dimy, int dimz){
  m_dimx = dimx;
  m_dimy = dimy;
  m_dimz = dimz;
  m_scheduledWalls = std::vector< std::vector<int> >();


};


void Sim3DCPU::init(){
  int dimx = m_dimx;
  int dimy = m_dimy;
  int dimz = m_dimz;
  
  const int         outerDimx  = dimx + 2 * RADIUS;
  const int         outerDimy  = dimy + 2 * RADIUS;
  const int         outerDimz  = dimz + 2 * RADIUS;
  const size_t      volumeSize = outerDimx * outerDimy * outerDimz;

  m_volumeSize = volumeSize;

  dim3              dimBlock;
  dim3              dimGrid;

  m_cells = std::vector<Cell>(volumeSize);

  for(int i = 0; i < volumeSize; i++){
    m_cells[i] = Cell();
  }

  listen = Coord();
  pBore = Coord();

  m_audioBuffer = std::vector<float>(MAX_AUDIO_SIZE);
  
  m_i = 0;
  reset();

};


void Sim3DCPU::clean(){
}

Cell Sim3DCPU::DAT(int x, int y, int z){
  return m_cells[getGlobalIndex(x, y, z)];
}

float Sim3DCPU::P(int x, int y, int z){
  //this is NEW STATE, cuz velocity step uses new
  return DAT(x,y,z).newState.pressure;
}

Velocity Sim3DCPU::V(int x, int y, int z){
  return DAT(x,y,z).oldState.velocity;
}

float Sim3DCPU::SIG(int x, int y, int z){
  return DAT(x,y,z).sigma * (0.5 / m_dt / PML);
}

float Sim3DCPU::SPRIME(int x, int y, int z){
  return 1 - BETA(x,y,z) + SIG(x,y,z);
}

int Sim3DCPU::BETA(int x, int y, int z){
  return DAT(x,y,z).beta();
}

float Sim3DCPU::vStep(int x, int y, int z, Velocity vb, int axis, float gradCoeff){
  Velocity v = V(x, y, z);

  int dx = axis == 0 ? 1 : 0;
  int dy = axis == 1 ? 1 : 0;
  int dz = axis == 2 ? 1 : 0;

  int beta = std::min(BETA(x, y, z), BETA(x + dx, y + dy, z + dz));
  float sigprime = 1 - beta + SIG(x, y, z);

  float v_num = beta * v[axis] - beta * gradCoeff * ( P(x + dx, y + dy, z + dz) - P(x, y, z) ) + sigprime * m_dt * vb[axis];
  float v_den = beta + sigprime * m_dt;
  // if(vb.z != 0 && axis == 2){
  //   printf("vb.z %f \n", vb[axis]);
  //   printf("beta %d \n", beta);
  //   printf("sig %f \n", SIG(x,y,z));
  //   printf("sigprime %f \n", sigprime);
  //   printf("v_num %f \n", v_num);
  //   printf("v_den %f \n", v_den);
  //   printf("v %f \n", v_num/v_den);

  // }
  return v_num / v_den;
}

void Sim3DCPU::step(int n){
  int onePercentStep = n/100;
  for(int i = 0; i < n; i++){
    //TODO swap
    float divCoeff = RHO * Cs * Cs * m_dt / m_ds;
    float gradCoeff = m_dt / m_ds / RHO;
    for(int z = 0; z < m_dimz; z++){
      for(int y = 0; y < m_dimy; y++){
        for(int x = 0; x < m_dimx; x++){
          float divStencil = 
            V(x,y,z).x - V(x-1,y,z).x
          + V(x,y,z).y - V(x,y-1,z).y
          + V(x,y,z).z - V(x,y,z-1).z;

          

          float p = DAT(x,y,z).oldState.pressure;
          float p_next =(p - divCoeff * divStencil)/(1 + SPRIME(x,y,z)*m_dt);

          // if(divStencil != 0){
          //   printf("OLD P %f\n", p);
          //   printf("NEW P %f\n", p_next);
          //   printf("DIV STENCIL %f\n", divStencil);
          //   printf("DIV COEFF %f\n", divCoeff);
          //   printf("DENOM %f\n", 1 + SPRIME(x,y,z)*m_dt);
          // }
          
          m_cells[getGlobalIndex(x,y,z)].newState.pressure = p_next;
        }
      }
    }
    for(int z = 0; z < m_dimz; z++){
      for(int y = 0; y < m_dimy; y++){
        for(int x = 0; x < m_dimx; x++){

          Velocity vb;
          vb.x = 0;
          vb.y = 0;
          vb.z = 0;
          if(DAT(x,y,z).isExcitor){
            

            float delta_p = std::max(m_p_mouth - DAT(pBore.x, pBore.y, pBore.z).oldState.pressure, 0.0f);
            //float delta_p_mod = max(0.05 * DELTA_P_MAX, delta_p);            
            //float shelf = 0.5 + 0.5 * tanh(4 * (-1 + (DELTA_P_MAX - delta_p_mod)/(0.01 * DELTA_P_MAX))); //unclear if DELTA_P_MAX is in the denominator...
            // float shelf = 1;
            
            float u_bore = W_J * H_R * std::max((1 - delta_p / DELTA_P_MAX), 0.0) * sqrt(2 * delta_p / RHO) ;
            float excitation = u_bore / (m_ds * m_ds * 50 * m_numExcitors); //hashtag units!  
            vb.z = excitation;

            // // printf("IS EXCITOR\n");
            // printf("p_mouth %f\n", m_p_mouth);
            // printf("p_bore %d %d %d\n", pBore.x, pBore.y, pBore.z);
            // printf("Excitor %d %d %d\n", x, y, z);

            // printf("p_bore %f\n", DAT(pBore.x, pBore.y, pBore.z).oldState.pressure);

            // // printf("num_excitors %d\n", m_numExcitors);
            // printf("u_bore %f\n", u_bore);
            // printf("excitation %f\n", excitation);

          }

          // int beta_x = std::min(BETA(x, y, z), BETA(x + 1, y, z));
          // int sigprime_x = 1 - beta_x + SIG(x, y, z);
          // float vx_num = beta_x * v - beta_x * gradCoeff * ( P(x + 1, y, z).x - P(x, y, z).x ) + sigprime_x * m_dt * vb.x;
          // float vx_den = beta_x + sigprime_x * m_dt;

          m_cells[getGlobalIndex(x,y,z)].newState.velocity.x = vStep(x, y, z, vb, 0, gradCoeff);
          m_cells[getGlobalIndex(x,y,z)].newState.velocity.y = vStep(x, y, z, vb, 1, gradCoeff);
          m_cells[getGlobalIndex(x,y,z)].newState.velocity.z = vStep(x, y, z, vb, 2, gradCoeff);
        }
      }
    }

    //swap
    for(int z = 0; z < m_dimz; z++){
      for(int y = 0; y < m_dimy; y++){
        for(int x = 0; x < m_dimx; x++){
          m_cells[getGlobalIndex(x,y,z)].oldState = m_cells[getGlobalIndex(x,y,z)].newState;

          // if(m_cells[getGlobalIndex(x,y,z)].oldState.velocity.z != 0){
          //   printf("VEL %f\n", m_cells[getGlobalIndex(x,y,z)].oldState.velocity.z);
          // }
        }
      }
    }


    m_audioBuffer[m_i] = P(listen.x,listen.y,listen.z);
    m_i += 1;

    if(onePercentStep > 0){
      if(i % onePercentStep == 0){
        printf("CPU %d/100\n", i/onePercentStep);
      }
    }
    else{
      printf("CPU %d\n", i);
    }
  }
}



void Sim3DCPU::setDT(float val){
  m_dt = val;
};
void Sim3DCPU::setDS(float val){
  m_ds = val;
};
void Sim3DCPU::setZN(float val){
  m_zn = val;
};

std::vector<float> Sim3DCPU::readBackAudio(){
  
  int N = m_i/OVERSAMPLE;
  std::vector<float> v(N);
  for(int i = 0; i < N; i++){
    v[i] = m_audioBuffer[i * OVERSAMPLE];
  }

  return v;
};

std::vector< std::vector<float> > Sim3DCPU::readBackData(){
  std::vector< std::vector<float> > v(m_volumeSize);
  for(int i = 0; i < m_volumeSize; i++){
    std::vector<float> vec;
    State output = m_cells[i].oldState;
    vec.push_back(output.pressure);
    vec.push_back(output.velocity.x);
    vec.push_back(output.velocity.y);
    vec.push_back(output.velocity.z);
    // if(output.pressure != 0){
    //   printf("pressure %f\n", output.pressure);

    // }

    v[i] = vec;
  }

  return v;
};

std::vector< std::vector<float> > Sim3DCPU::readBackDataCoords(std::vector< std::vector<int> > coords){
  std::vector< std::vector<float> > v = readBackData();
  std::vector< std::vector<float> > data;
  for(int i = 0; i < coords.size(); i++){
    data.push_back(v[getGlobalIndex(coords[i][0], coords[i][1], coords[i][2])]);
  }
  return data;
};


std::vector<int> Sim3DCPU::readBackAux(){
  std::vector<int> v(m_volumeSize);
  for(int i = 0; i < m_volumeSize; i++){
    v[i] = m_cells[i].beta() << BETA_SHIFT;
  }
  return v;
};

int Sim3DCPU::getStrideY(){
  return m_dimx + 2 * RADIUS;
}

int Sim3DCPU::getStrideZ(){
  return getStrideY() * (m_dimy +  2 * RADIUS);
}

int Sim3DCPU::getGlobalIndex(int x, int y, int z){
  int startOfInnerVolume = 
    RADIUS 
  + RADIUS * getStrideY()  
  + RADIUS * getStrideZ();

  
  int target = startOfInnerVolume + 
    x 
  + y * getStrideY() 
  + z * getStrideZ();
  return target;
}



void Sim3DCPU::setSigma(int x, int y, int z, int level){
  int i = getGlobalIndex(x, y, z);
  m_cells[i].sigma = level;
}

void Sim3DCPU::setWall(int x, int y, int z, int val){
  int g = getGlobalIndex(x, y, z);
  m_cells[g].isWall = val == 0;
}


void Sim3DCPU::setExcitors(std::vector< std::vector<int> > excitors){
  m_numExcitors = 0;
  for(int i = 0; i < excitors.size(); i++){
    int x = excitors[i][0];
    int y = excitors[i][1];
    int z = excitors[i][2];
    int val = excitors[i][3];
    int loc = getGlobalIndex(x, y, z);
    m_cells[loc].isExcitor = val == 1;
    if(m_cells[loc].isExcitor){
      m_numExcitors += 1;
    }
  }
};

void Sim3DCPU::setListener(int x_l, int y_l, int z_l){
  // m_i_global_listener = getGlobalIndex(x_l, y_l, z_l);
  listen = Coord();
  listen.x = x_l;
  listen.y = y_l;
  listen.z = z_l;
}

void Sim3DCPU::setPBore(int x_l, int y_l, int z_l){
  // m_i_global_p_bore = getGlobalIndex(x_l, y_l, z_l);
  pBore = Coord();
  pBore.x = x_l;
  pBore.y = y_l;
  pBore.z = z_l;
}

void Sim3DCPU::setPressureMouth(float pressure){
  m_p_mouth = pressure;
}

void Sim3DCPU::setExcitor(int x, int y, int z, int val){
  int loc = getGlobalIndex(x, y, z);
  m_cells[loc].isExcitor = val == 1;
}

void Sim3DCPU::restart(){
  m_i = 0;
  
  for(int i = 0; i < m_cells.size(); i++){
    m_cells[i].oldState.pressure = 0;
    m_cells[i].oldState.velocity.x = 0;
    m_cells[i].oldState.velocity.y = 0;
    m_cells[i].oldState.velocity.z = 0;

    m_cells[i].newState.pressure = 0;
    m_cells[i].newState.velocity.x = 0;
    m_cells[i].newState.velocity.y = 0;
    m_cells[i].newState.velocity.z = 0;
  }

}
void Sim3DCPU::reset(){
  // Toggle the buffers
  // Visual Studio 2005 does not like std::swap
  //    std::swap<float *>(bufferSrc, bufferDst);
  m_i = 0;
  for(int i = 0; i < m_cells.size(); i++){
    m_cells[i].oldState.pressure = 0;
    m_cells[i].oldState.velocity.x = 0;
    m_cells[i].oldState.velocity.y = 0;
    m_cells[i].oldState.velocity.z = 0;

    m_cells[i].newState.pressure = 0;
    m_cells[i].newState.velocity.x = 0;
    m_cells[i].newState.velocity.y = 0;
    m_cells[i].newState.velocity.z = 0;
    m_cells[i].isWall = false;
    m_cells[i].isExcitor = false;
    m_cells[i].sigma = 0;
  }

  for(int i = 0; i < PML + 1; i++){
    for(int x = i; x < m_dimx - i; x++){
      for(int y = i; y < m_dimy - i; y++){
        for(int z = i; z < m_dimz - i; z++){
          setSigma(x, y, z, PML - i);
        }
      }
    }
  }

}

void Sim3DCPU::scheduleWall(int x, int y, int z, int val){
  std::vector<int> v;
  v.push_back(x);
  v.push_back(y);
  v.push_back(z);
  v.push_back(val);

  m_scheduledWalls.push_back(v);
}

void Sim3DCPU::writeWalls(){

  for(int i = 0; i < m_scheduledWalls.size(); i++){
    int x = m_scheduledWalls[i][0];
    int y = m_scheduledWalls[i][1];
    int z = m_scheduledWalls[i][2];
    int val = m_scheduledWalls[i][3];
    int g = getGlobalIndex(x, y, z);
    m_cells[g].isWall = val == 0;
  }

}
