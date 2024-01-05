index = 1:7;
data_type = ["dtheta", "dspeed"];
lengths = [200, 400, 600];
MIP_CUT = cell(2, 7);
MIP_PHI = cell(2, 7);

for i = 1:7
    MIP_CUT{1, i} = readmatrix("MIP_CUT_" + data_type(1) + "_" + num2str(index(i)) ...
        + "_" + num2str(lengths(l)) + "_20 (sec).csv");
    MIP_CUT{2, i} = readmatrix("MIP_CUT_" + data_type(2) + "_" + num2str(index(i)) ...
        + "_" + num2str(lengths(l)) + "_20 (sec).csv");
    MIP_PHI{1, i} = readmatrix("PHIs_" + data_type(1) + "_" + num2str(index(i)) ...
        + "_" + num2str(lengths(l)) + "_20 (sec).csv");
    MIP_PHI{2, i} = readmatrix("PHIs_" + data_type(2) + "_" + num2str(index(i)) ...
        + "_" + num2str(lengths(l)) + "_20 (sec).csv");    
end