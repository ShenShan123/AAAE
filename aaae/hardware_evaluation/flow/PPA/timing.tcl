# 1. 读取工艺库
read_liberty @AAAE_PPA_RESOURCES@/NangateOpenCellLibrary.lib

# 2. 读取设计网表（必须是 yosys 或其他工具生成的 netlist）
read_verilog ./synth.v

# 3. 设置顶层模块

link_design top
set_wire_load_model -name "5K_hvratio_1_1" -library "NangateOpenCellLibrary"

# 4. 读取 SDC 约束（可以使用 Yosys 生成的，或者手动写）
read_sdc @AAAE_FLOW_ROOT@/PPA/constraint.sdc

# 5. 报告关键路径
report_checks

#report_power
write_sdf ./timing.sdf


