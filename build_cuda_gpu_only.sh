#sleep so files can be udpated in time
ssh larry@csiga.csail.mit.edu 'sleep 1; make -C ~/code/CPP/gpu/sim/ clean; make -C ~/code/CPP/gpu/sim/ libSim.a;'
