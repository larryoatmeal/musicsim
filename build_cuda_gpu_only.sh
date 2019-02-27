#sleep so files can be udpated in time
ssh larryw@35.226.198.237 'sleep 1; make -C ~/code/CPP/gpu/sim/ clean; make -C ~/code/CPP/gpu/sim/ libSim.a;'
