# Build the Basys 3 bitstream from the command line (no Vivado project needed).
#
#   vivado -mode batch -source scripts/build.tcl                      (fast engine, the default)
#   vivado -mode batch -source scripts/build.tcl -tclargs systolic    (systolic-array engine)
#
# Outputs go to build/<engine>/ (git-ignored): top.bit plus utilization, timing, DRC
# and power reports. Paths are worked out from this script's location, so it runs
# from any folder on any machine.

set ENGINE [expr {[llength $argv] > 0 ? [lindex $argv 0] : "fast"}]
if {$ENGINE ni {fast systolic}} {
    error "engine must be fast or systolic, got '$ENGINE'"
}
set FAST  [expr {$ENGINE eq "fast" ? 1 : 0}]

set REPO  [file normalize [file join [file dirname [info script]] ..]]
set OUT   $REPO/build/$ENGINE
set PART  xc7a35tcpg236-1
file mkdir $OUT

read_verilog -sv [list \
    $REPO/rtl/pe.sv \
    $REPO/rtl/systolic_array.sv \
    $REPO/rtl/dense.sv \
    $REPO/rtl/mlp.sv \
    $REPO/rtl/adder_tree.sv \
    $REPO/rtl/fast_mlp.sv \
    $REPO/rtl/uart_rx.sv \
    $REPO/rtl/uart_tx.sv \
    $REPO/rtl/top.sv]
read_xdc $REPO/constraints/basys3.xdc

synth_design -top top -part $PART -generic FAST=$FAST -generic "HEX_DIR=\"$REPO/model/weights/hex/\""
opt_design
place_design
phys_opt_design
route_design

report_utilization    -file $OUT/utilization.rpt
report_timing_summary -file $OUT/timing.rpt
report_drc            -file $OUT/drc.rpt
report_power          -file $OUT/power.rpt
write_bitstream -force $OUT/top.bit

puts "Bitstream ($ENGINE engine) written to $OUT/top.bit"
