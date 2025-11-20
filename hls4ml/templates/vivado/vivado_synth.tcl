set tcldir [file dirname [info script]]
source [file join $tcldir project.tcl]

add_files ${project_name}_prj/solution1/syn/vhdl
read_xdc constraints.xdc
synth_design -top ${project_name} -part $part
report_utilization -file vivado_synth.rpt
write_verilog -force ${project_name}_netlist.v
report_operating_conditions -all

# Set specific device and environment operating conditions
set_operating_conditions -ambient 25

#---------------------- Power estimation at post synthesis ----------------

# Generate verbose post-synthesis power report
report_power -verbose -file ex1_post-synthesis.pwr

#----Run various Implementation steps then run Power estimation after every step ----
opt_design
report_power -verbose -file ex1_post-opt_design.pwr
power_opt_design ;# Optional
report_power -verbose -file ex1_post_pwr_opt_design.pwr
place_design
report_power -verbose -file ex1_post_place_design.pwr
phys_opt_design ;# Optional
report_power -verbose -file ex1_post_phys_opt_design.pwr
route_design

# Generate post-route verbose power report
report_power -verbose -file ex1_post_route_design.pwr

# Return operating conditions to default for device
reset_operating_conditions -ambient -voltage {vccint vccaux}