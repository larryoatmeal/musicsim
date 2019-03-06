#sleep so files can be udpated in time
ssh larry@csiga.csail.mit.edu 'sleep 1; make -C /media/ssd1/larry/code/CPP/gpu/sim/ clean; make -C /media/ssd1/larry/code/CPP/gpu/sim/ libSim.a; make -C /media/ssd1/larry/code/CPP/cpu_only/build pytest;'
