module adder_15bit (
    input [14:0] a,    // 输入a，15位
    input [14:0] b,    // 输入b，15位
	input clk,
    output [15:0] sum  // 输出sum，16位
);
    wire [14:0] carry; // 中间进位
    assign sum[0] = a[0];
    // 实例化第一个全加器，cin设为0
    Half_adder ha0 (
        .a(a[1]),
        .b(b[1]),
        .S(sum[1]),
        .C(carry[1])
    );

    // 实例化后续的14个全加器
    genvar i;
    generate
        for (i = 2; i < 15; i = i + 1) begin : full_adders
            Full_adder fa (
                .a(a[i]),
                .b(b[i]),
                .CIN(carry[i-1]),
                .S(sum[i]),
                .C(carry[i])
            );
        end
    endgenerate
	assign sum[15]=carry[14];
`ifdef SIM
    // 最后的进位
integer Half_adder_0_3;
integer Full_adder_1_3;
integer Full_adder_2_3;
integer Full_adder_3_3;
integer Full_adder_4_3;
integer Full_adder_5_3;
integer Full_adder_6_3;
integer Full_adder_7_3;
integer Full_adder_8_3;
integer Full_adder_9_3;
integer Full_adder_10_3;
integer Full_adder_11_3;
integer Full_adder_12_3;
integer Full_adder_13_3;
//integer Full_adder_14_3;

initial begin
    Half_adder_0_3 = $fopen("3_Half_adder_0.txt","w");
	Full_adder_1_3 = $fopen("3_Full_adder_1.txt","w");
	Full_adder_2_3 = $fopen("3_Full_adder_2.txt","w");
	Full_adder_3_3 = $fopen("3_Full_adder_3.txt","w");
	Full_adder_4_3 = $fopen("3_Full_adder_4.txt","w");
	Full_adder_5_3 = $fopen("3_Full_adder_5.txt","w");
	Full_adder_6_3 = $fopen("3_Full_adder_6.txt","w");
	Full_adder_7_3 = $fopen("3_Full_adder_7.txt","w");
	Full_adder_8_3 = $fopen("3_Full_adder_8.txt","w");
	Full_adder_9_3 = $fopen("3_Full_adder_9.txt","w");
	Full_adder_10_3 = $fopen("3_Full_adder_10.txt","w");
	Full_adder_11_3 = $fopen("3_Full_adder_11.txt","w");
    Full_adder_12_3 = $fopen("3_Full_adder_12.txt","w");
    Full_adder_13_3 = $fopen("3_Full_adder_13.txt","w");
end
always @(posedge clk) begin
    $fdisplay(Half_adder_0_3, "%b",{a[1],b[1]});
	$fdisplay(Full_adder_1_3, "%b",{a[2],b[2],carry[1]});
	$fdisplay(Full_adder_2_3, "%b",{a[3],b[3],carry[2]});
	$fdisplay(Full_adder_3_3, "%b",{a[4],b[4],carry[3]});
	$fdisplay(Full_adder_4_3, "%b",{a[5],b[5],carry[4]});
	$fdisplay(Full_adder_5_3, "%b",{a[6],b[6],carry[5]});
	$fdisplay(Full_adder_6_3, "%b",{a[7],b[7],carry[6]});
	$fdisplay(Full_adder_7_3, "%b",{a[8],b[8],carry[7]});
	$fdisplay(Full_adder_8_3, "%b",{a[9],b[9],carry[8]});
	$fdisplay(Full_adder_9_3, "%b",{a[10],b[10],carry[9]});
	$fdisplay(Full_adder_10_3, "%b",{a[11],b[11],carry[10]});
	$fdisplay(Full_adder_11_3, "%b",{a[12],b[12],carry[11]});
    $fdisplay(Full_adder_12_3, "%b",{a[13],b[13],carry[12]});
    $fdisplay(Full_adder_13_3, "%b",{a[14],b[14],carry[13]});
end
`endif
endmodule


