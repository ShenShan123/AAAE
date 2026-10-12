.PHONY: com sim cov clean db debug

OUTPUT = dut
ALL_DEFINE = +define+DUMP_VPD+NTC+RECREM+SIM
#LIB
LIB_FILE = @AAAE_PPA_RESOURCES@/NangateOpenCellLibrary.v


#Code converage command
CM = -cm line+cond+fsm+branch+tgl
CM_NAME = -cm_name $(OUTPUT)
CN_DIR = -cm_dir ./$(OUTPUT).vdb

#vpd file name
VPD_NAME = +vpdfile+$(OUTPUT).vpd

#Compile command
VCS = vcs +v2k -timescale=1ns/1ps					\
		-full64										\
		-debug_acc+all								\
		-debug_region+cell+encrypt					\
		-sverilog									\
		-v $(LIB_FILE)								\
		+sdfverbose									\
		$(CM)										\
		$(CM_NAME)									\
		$(CM_DIR)									\
		$(ALL_DEFINE)								\
		$(VPD_NAME)									\
		-o $(OUTPUT)								\
		-l compile.log								\
		+neg_tchk									
#nospecify
#compsdf
#+vcs+flush+all
#neg_tchk

#simulation command
SIM = ./$(OUTPUT)									\
	  $(CM) $(CM_NAME) $(CM_DIR)					\
	  $(VPD_NAME)									\
	  $(ALL_DEFINE)									\
	  -l $(OUTPUT).log
#+ntb_ramdom_seend_automatic
#start compile
com:
	$(VCS) -f filelist.f 

#start simulation
sim:
	$(SIM)

#show the coverage
cov:
	dve -covdir *vdb &

debug:
	dve -vpd	$(OUTPUT).vpd &
#verdi debug
db:
	verdi -sv -f file_list.f -ssf $(OUTPUT).fsdb
#start clean
clean:
	rm -rf ./csrc *.daidir simv* *.sdf *.key *.vpd ./DVEfiles ${OUTPUT} *.vdb *.fsdb *.vcd *.txt sdfAnnotateInfo synth.v
