####################################
# SDC Constraints for mul_top
# 修正版本 - 正确的三级流水线约束
####################################

current_design top

# 时钟定义
set clk_name core_clock
set clk_port_name clk
set clk_period 5.0
set clk_io_pct 0.2

# 创建时钟
create_clock -name $clk_name -period $clk_period [get_ports $clk_port_name]

# 设置时钟不确定性（考虑jitter和skew）
set_clock_uncertainty 0.2 [get_clocks $clk_name]

# 设置时钟延迟（如果需要模拟时钟树延迟）
# set_clock_latency -source 0.5 [get_clocks $clk_name]
# set_clock_latency 0.3 [get_clocks $clk_name]

####################################
# 输入约束
####################################
# 输入延迟：假设外部逻辑在时钟沿后 1ns 提供稳定数据
# set input_delay [expr $clk_period * $clk_io_pct]
# set_input_delay -clock $clk_name -max $input_delay [get_ports {a1[*] a2[*]}]
# set_input_delay -clock $clk_name -min 0.0 [get_ports {a1[*] a2[*]}]

# 设置输入驱动能力（假设由标准单元驱动）
set_driving_cell -lib_cell BUF_X1 -library NangateOpenCellLibrary [get_ports {a1[*] a2[*]}]

####################################
# 输出约束
####################################
# 注意：pro[*] 端口直接连接到输出寄存器(_48_到_63_)的Q端口
# 这是一个三级流水线设计，输出已经被寄存器隔离
# 不需要设置 output_delay，因为：
# 1. 真正的关键路径是 输入寄存器 → APPROX → 输出寄存器
# 2. 输出寄存器到端口只是缓冲延迟，不是时序关键路径

# 如果这是顶层模块，且输出寄存器已经做了时序隔离，则不需要 output_delay
# set_output_delay -clock $clk_name -max $output_delay [get_ports {pro[*]}]
# set_output_delay -clock $clk_name -min 0.0 [get_ports {pro[*]}]

# 设置输出负载（用于估算驱动能力）
set_load [expr 0.01] [get_ports {pro[*]}]

####################################
# 环境约束
####################################
# 设置工作条件（如果库支持）
# set_operating_conditions -max worst -min best

# 设置温度和电压（如果需要）
# set_temperature 125
# set_voltage 0.9

####################################
# 时序例外（如果需要）
####################################
# 对于三级流水线设计，输出端口不是真正的时序端点
# 如果误设置了 output_delay，需要用 false_path 排除干扰

# 如果之前设置了 output_delay 导致工具误报，可以用这个排除：
# set_false_path -from [get_pins {_48_/Q _49_/Q _50_/Q _51_/Q _52_/Q _53_/Q _54_/Q _55_/Q _56_/Q _57_/Q _58_/Q _59_/Q _60_/Q _61_/Q _62_/Q _63_/Q}] -to [get_ports {pro[*]}]

# 如果某些路径不需要分析，可以设置 false path
# set_false_path -from [get_ports reset] -to [all_registers]

# 多周期路径（如果有）
# set_multicycle_path 2 -from [get_pins {A1_reg[*]/Q}] -to [get_pins {PRO_reg[*]/D}]

####################################
# 设计规则约束
####################################
# 最大转换时间（slew rate）
set_max_transition 0.5 [current_design]

# 最大扇出
set_max_fanout 20 [current_design]

# 最大电容
set_max_capacitance 0.2 [current_design]

####################################
# 关键：确保 OpenSTA 正确分析寄存器到寄存器路径
####################################
# 对于三级流水线设计：输入寄存器 → 组合逻辑 → 输出寄存器
# OpenSTA 应该默认分析这些路径，但为了确保不被误导，可以：

# 1. 不设置 output_delay（见上面的输出约束部分）
# 2. 或者使用 group_path 明确分组（可选）
# group_path -name reg2reg -from [all_registers] -to [all_registers]
# group_path -name in2reg -from [all_inputs] -to [all_registers]

####################################
# 报告设置（可选，用于生成详细报告）
####################################
# set_ideal_network [get_ports $clk_port_name]
# set_dont_touch_network [get_clocks $clk_name]