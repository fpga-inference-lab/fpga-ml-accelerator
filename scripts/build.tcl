# Build the Basys 3 bitstream from the command line (no Vivado project needed).
#
#   vivado -mode batch -source scripts/build.tcl
#
# Outputs go to build/ (git-ignored): top.bit plus utilization, timing and DRC reports.
# Paths are worked out from this script's location, so it runs from any folder on any machine.

set REPO  [file normalize [file join [file dirname [info script]] ..]]
set OUT   $REPO/build
set PART  xc7a35tcpg236-1
file mkdir $OUT

read_verilog -sv [list \
    $REPO/rtl/pe.sv \
    $REPO/rtl/systolic_array.sv \
    $REPO/rtl/dense.sv \
    $REPO/rtl/mlp.sv \
    $REPO/rtl/uart_rx.sv \
    $REPO/rtl/uart_tx.sv \
    $REPO/rtl/top.sv]
read_xdc $REPO/constraints/basys3.xdc

synth_design -top top -part $PART -generic "HEX_DIR=\"$REPO/model/weights/hex/\""
opt_design
place_design
route_design

report_utilization    -file $OUT/utilization.rpt
report_timing_summary -file $OUT/timing.rpt
report_drc            -file $OUT/drc.rpt
write_bitstream -force $OUT/top.bit

puts "Bitstream written to $OUT/top.bit"
