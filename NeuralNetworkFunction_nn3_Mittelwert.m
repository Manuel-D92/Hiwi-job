function [y1] = NeuralNetworkFunction_nn3_Mittelwert(x1)
%MYNEURALNETWORKFUNCTION neural network simulation function.
%
% Auto-generated by MATLAB, 09-Sep-2019 11:54:49.
%
% [y1] = myNeuralNetworkFunction(x1) takes these arguments:
%   x = 3xQ matrix, input #1
% and returns:
%   y = 1xQ matrix, output #1
% where Q is the number of samples.

%#ok<*RPMT0>

% ===== NEURAL NETWORK CONSTANTS =====

% Input 1
x1_step1.xoffset = [-400;0;-5567.93333333333];
x1_step1.gain = [0.00333333333333333;0.133333333333333;8.15607464439514e-05];
x1_step1.ymin = -1;

% Layer 1
b1 = [-3.975805295997169253;3.6026497384355020692;-0.24454353709019491236;-0.88835949773777123095;7.2099649249663615436;1.2887543354110391203;-0.55839754142992803487;0.22562570656837888095;-3.3166003248703543882;2.4612902708603141555;-1.4631305460233321458;-2.0516515329758586716;-8.4269562468222805052;1.9341899900523242817;0.095494280453136348719;7.2423814097195764461;1.4459090358821724553;-1.2582469774941795748];
IW1_1 = [5.3891390936138119372 0.072793162851480130993 4.1556605510663287006;2.2440306493684682998 -2.4289096447757692054 -0.41279014048632595157;0.88691098595167727758 0.78883964093067526768 -2.1040093536865884793;0.30198786570117369665 0.50103491414456602371 1.7650528517425776442;9.5761436943415390033 17.675099051548464502 -13.276502712337448386;1.4624572046621706622 -0.70152208553770811772 -2.0848742348564353044;-0.18437609542257030193 2.8076171408286847431 -4.9555457213162235064;-0.023381165078670100577 -1.3160438924439794572 3.0765510263597599661;3.0559102498369430378 5.5267788008737008809 -5.0918887068068530155;-0.086412029633144524232 0.067047374504195414446 7.7116596384431899125;-1.5674245058137261655 -2.5626860325138087759 0.1909463694356975938;-2.2871566000684238062 -3.7089705180527654527 -0.3427150543602139332;-1.2145994456574173981 -2.514539616145595069 -13.097395989048246534;1.0718502701123526766 -0.55107766608421959731 -0.52273087905572823697;-0.1029353151034337055 -0.83285578669117887873 2.9644556927125753631;9.7292836950296965881 18.219074057560710855 -13.557107252315296009;2.9948640774964081501 0.63510795133445308025 -0.80883915296399355732;-0.029805824906884374748 -0.52054239308822658483 2.7143143284816328986];

% Layer 2
b2 = 0.12009065524363841948;
LW2_1 = [0.015218069918743479599 -0.29839015332142360126 1.1046308326783795817 -0.92517214072869669028 5.3107652471268425032 -0.082589952723127482703 -0.58705942644669018016 -4.0598336147852993605 -0.1164673480104501252 0.3579744695542923516 3.4699762802816551677 -1.515821284230410404 -0.82553598509058601085 1.4803966471411962313 4.4604465531656325439 -5.0450584614173710207 -0.15532658180108319113 1.5869400838966181144];

% Output 1
y1_step1.ymin = -1;
y1_step1.gain = 0.005;
y1_step1.xoffset = 0;

% ===== SIMULATION ========

% Dimensions
Q = size(x1,2); % samples

% Input 1
xp1 = mapminmax_apply(x1,x1_step1);

% Layer 1
a1 = tansig_apply(repmat(b1,1,Q) + IW1_1*xp1);

% Layer 2
a2 = repmat(b2,1,Q) + LW2_1*a1;

% Output 1
y1 = mapminmax_reverse(a2,y1_step1);
end

% ===== MODULE FUNCTIONS ========

% Map Minimum and Maximum Input Processing Function
function y = mapminmax_apply(x,settings)
y = bsxfun(@minus,x,settings.xoffset);
y = bsxfun(@times,y,settings.gain);
y = bsxfun(@plus,y,settings.ymin);
end

% Sigmoid Symmetric Transfer Function
function a = tansig_apply(n,~)
a = 2 ./ (1 + exp(-2*n)) - 1;
end

% Map Minimum and Maximum Output Reverse-Processing Function
function x = mapminmax_reverse(y,settings)
x = bsxfun(@minus,y,settings.ymin);
x = bsxfun(@rdivide,x,settings.gain);
x = bsxfun(@plus,x,settings.xoffset);
end
