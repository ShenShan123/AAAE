-timescale=1ns/1ps
//Macro define
//+define+INC_COUNTER

//Source file
./synth.v
@AAAE_PPA_RESOURCES@/NangateOpenCellLibrary.v
./pthread_yield_fix.o

//Testbench
./Testbench.v
