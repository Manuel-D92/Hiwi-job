% Beispielprogramm zum Einlesen von Messdaten 

% Löschen des Workspace 
clear all 
% Alle Plot-Fenster schließen 
close all 
% Command Window löschen 
clc 

% Lade Messdaten-Datei (Roh-Daten)
load Measurement_2013_4_17_15_2_raw_unfixed.mat 

% Extrahiere Namen der Signale 
Namen = MeasurementTableRaw{1,1}; 

% Extrahiere Daten der Signale 
Daten = MeasurementTableRaw{2,1}; 
AnzahlSignale = size(Daten,2); 
AnzahlWerte   = size(Daten,1); 

% Die Spalten der Matrix "Daten" enthalten alle Signale. 
% Im Folgenden wird für alle Spalten je ein Signalvektor 
% mit dem entsprechenden Namen erzeugt 

for(k=1:AnzahlSignale)
    % Erstelle einen Matlab-Befehl, der eine Spalte aus der 
    % Datenmatrix in einen Vektor kopiert 
    Befehl = [Namen{1,k},'=Daten(:,k);']; 
    % Führe diesen Befehl aus 
    eval(Befehl); 
end

% Nun sind alle Signale eingelesen und können z. B. geplottet werden: 
% Beispielplot über alle Messungen 
%figure(1); 
kk = 1:AnzahlWerte; 
%plot(kk,i_d_ist,'r',kk,Id_ref,'k'); 
%title('Stromkomponente Id'); 
%xlabel('Messpunkte'); 
%ylabel('Id/A'); 
%legend('Istwert','Sollwert'); 

% Beispielplot, der Id, Iq und If darstellt
%figure(2)
%plot3(Id_ref,Iq_ref,Iexc_ref,'kx'); 
%title('Stromsollwerte im Datensatz'); 
%xlabel('Id\_ref/A'); 
%ylabel('Iq\_ref/A'); 
%zlabel('Iexc\_ref/A'); 
%zlim([0 30])
%grid on 
%box on 
%view(35,70)


%figure(3);
anzahl_mot =0;
anzahl_gen=0;
anzahl_mot_Mittelwert=0;


test = CAN_VES_M_HM1_Ist(3);


%M = {};
k=1;

AnzahlMessungen = 15;
Mref = 0;
% Mref berechnen mit Iexc_ref, Id_ref und Iq_ref

Ld       = 615e-6; 
Lq       = 360e-6; 
Lm       = 0.016; 

for(n=1:AnzahlWerte)
    
    Mref(n) = (3/2*4*Iq_ref(n)*(Iexc_ref(n)* Lm+(Ld-Lq)*Id_ref(n)));
    
end

% Mittelwert bilden
for(n=1: (AnzahlWerte/AnzahlMessungen))
    CAN_VES_M_Mittel=0;
    CAN_VES_n_Mittel=0;
    CAN_VES_U_Mittel=0;
    CAN_VES_I_Mittel=0;
    
    Iq_Mittel=0;
    Id_Mittel=0;
    If_Mittel=0;
    
    Id_ist_Mittel=0;
    Iq_ist_Mittel=0;
    
    for (m=1: AnzahlMessungen)
    

      CAN_VES_M_Mittel(m) = CAN_VES_M_HM1_Ist(m+((n-1)*AnzahlMessungen))/94; %94 = 9.4 getriebe übersetzung 10 Umrechnungsfaktor
      CAN_VES_n_Mittel(m) = CAN_VES_n_HM1_Ist(m+((n-1)*AnzahlMessungen));
      CAN_VES_U_Mittel(m) = CAN_VES_U_DC_Ist(m+((n-1)*AnzahlMessungen));
      CAN_VES_I_Mittel(m) = CAN_VES_I_DC_Ist(m+((n-1)*AnzahlMessungen));
      Mref_Mittel(m)= Mref(m+((n-1)*AnzahlMessungen));
      
      Iq_Mittel(m) = Iq_ref(m+((n-1)*AnzahlMessungen));
      Id_Mittel(m) = Id_ref(m+((n-1)*AnzahlMessungen));
      If_Mittel(m) = Iexc_ref(m+((n-1)*AnzahlMessungen));
      
      Id_ist_Mittel(m) = i_d_ist(m+((n-1)*AnzahlMessungen));
      Iq_ist_Mittel(m) = i_q_ist(m+((n-1)*AnzahlMessungen));
      Iexe_ist_Mittel(m) = Iexc_calibrated(m+((n-1)*AnzahlMessungen));
     
      Ud_Mittel(m)= u_d(m+((n-1)*AnzahlMessungen));
      Uq_Mittel(m)= u_q(m+((n-1)*AnzahlMessungen));
    end
    M_Mittelwert(n) = mean(CAN_VES_M_Mittel(:));
    n_Mittelwert(n) = mean(CAN_VES_n_Mittel(:));
    U_Mittelwert(n) = mean(CAN_VES_U_Mittel(:));
    I_Mittelwert(n) = mean(CAN_VES_I_Mittel(:));
    Mref_Mittelwert(n) = mean(Mref_Mittel(:));
    
    Iq_Mittelwert(n) = mean(Iq_Mittel(:));
    Id_Mittelwert(n) = mean(Id_Mittel(:));
    If_Mittelwert(n) = mean(If_Mittel(:));
    
    Id_ist_Mittelwert(n) = mean(Id_ist_Mittel(:));
    Iq_ist_Mittelwert(n) = mean(Iq_ist_Mittel(:));
    Iexe_ist_Mittelwert(n) = mean(Iexe_ist_Mittel(:));
    
    Ud_Mittelwert(n) = mean(Ud_Mittel(:));
    Uq_Mittelwert(n) = mean(Uq_Mittel(:));
end
% end Mittelwertbildung

%for (j=1 : AnzahlWerte/AnzahlMessungen)
  %if(CAN_VES_M_HM1_Ist(j)>0)
  %wirk_mot_Mittelwert(j) =((2 *pi *M_Mittelwert(j) *n_Mittelwert(j) /60) /(U_Mittelwert(j) *I_Mittelwert(j) ));
  %anzahl_mot_Mittelwert = anzahl_mot_Mittelwert +1;
  %is = [Id_ref(i),Iq_ref(i),Iexc_ref(i)];
  %Iq_Mittelwert(k) = (Iq_ref(j));
  %Id_Mittelwert(k) = (Id_ref(j));
  %If_Mittelwert(k) = (Iexc_ref(j));
  %M_Mittelwert(k) = (CAN_VES_M_Mittelwert(j));
  %k= k+1;
  %end
%end


for (i=1 : AnzahlWerte)
  
  if(CAN_VES_M_HM1_Ist(i)>0)
  wirk_mot(i) =((2 *pi *CAN_VES_M_HM1_Ist(i) *CAN_VES_n_HM1_Ist(i) /60) /(CAN_VES_U_DC_Ist(i) *CAN_VES_I_DC_Ist(i) ));
  anzahl_mot = anzahl_mot +1;
  %is = [Id_ref(i),Iq_ref(i),Iexc_ref(i)];
  Iq(k) = (Iq_ref(i));
  Id(k) = (Id_ref(i));
  If(k) = (Iexc_ref(i));
  M(k) = (CAN_VES_M_HM1_Ist(i));
  k= k+1;
end

  if(CAN_VES_M_HM1_Ist(i)<0)
  wirk_gen(i) = ((CAN_VES_U_DC_Ist(i) *CAN_VES_I_DC_Ist(i) ) /(2 *pi *CAN_VES_M_HM1_Ist(i) *CAN_VES_n_HM1_Ist(i) /60));
  anzahl_gen = anzahl_gen +1;
end
 
%wirk(i)  = sort((2 *pi *CAN_VES_M_HM1_Ist(i) *CAN_VES_n_HM1_Ist(i) /60) /(CAN_VES_U_DC_Ist(i) *CAN_VES_I_DC_Ist(i) ));

%wirkung = ((2 .*pi .*CAN_VES_M_HM1_Ist .*CAN_VES_n_HM1_Ist) ./(u_exc .*Iexc_ref .*3.6))

%Udc =
%Idc = 
end



max_in = max(wirk_mot);
[x_max,y_max]=find(wirk_mot==max_in);
Data_max_wirk(1,:) = Namen;
%Data_max_wirk(2,:) = Daten(y_max,:);
%Data_max_wirk(2,1)

wirk_mot (wirk_mot==0)= [];
%plot(0:(anzahl_mot-1),sort(wirk_mot),'r'); 

wirk_gen (wirk_gen==0)= [];
wirk_gen (wirk_gen<0)= [];

wirk_gen = sort(wirk_gen);


%figure(4);
%plot(0:(length(wirk_gen)-1),sort(wirk_gen),'b');

I = [Iq;Id;If]; % für motorisch rechenbarer Wirkungsgrad
%I = [Iq,Id,If];

I_Mittelwert = [Iq_Mittelwert;Id_Mittelwert;If_Mittelwert];

w= n_Mittelwert*2*pi;
Eingang = [Iq_Mittelwert;Id_Mittelwert;If_Mittelwert;w];
Ausgang = [Ud_Mittelwert;Uq_Mittelwert];

Eingang_nn3_Mittelwert = [Id_Mittelwert;If_Mittelwert;M_Mittelwert];
Eingang_nn3_alle  = [Id_ref';Iexc_ref';(CAN_VES_M_HM1_Ist/94)'];
Ausgang_nn3_Mittelwert = [Iq_Mittelwert];
Ausgang_nn3_alle = [Iq_ref'];


%Eingang_nn_Mref = [Id_ref/max(abs(Id));Iexc_ref/max(abs(If));Mref/max(abs(M))];
%Ausgang_nn_Mref = [Iq_ref/max(abs(Iq))];

Eingang_nn1_Mittelwert = [Iq_ist_Mittelwert;Id_ist_Mittelwert;Iexe_ist_Mittelwert];
Ausgang_nn1_Mittelwert = [M_Mittelwert];

Eingang_nn3 = [Id/max(abs(Id));If/max(abs(If));M/max(abs(M))];
Ausgang_nn3 = [Iq/max(abs(Iq))];

Eingang_nn33_Mittelwert = [Id_ist_Mittelwert;Iexe_ist_Mittelwert;Mref_Mittelwert];
Ausgang_nn33_Mittelwert = [Iq_ist_Mittelwert];


Messbeispiel_1 = [69.4593;-393.9231;5;-450.533]; % [Iq;Id;Ie;MSoll]
Messbeispiel_2 = [136.8081;-375.877;0;-3117];
Messbeispiel_3 = [57.735;-33.333;10;2680];
Messbeispiel_4 = [200;-346.4102;2.5;-2716];
Messbeispiel_5 = [57.350;33.333;15;2680];
Messbeispiel_6 = [257.115;-306.4178;10;9018];
Messbeispiel_7 = [375.877;-136.8081;15;18795];

E=Daten(:,[2 3 6]);
A=Daten(:,1);

%% plot der Ströme und dem Drehmoment
figure(1)
subplot(3,1,1);
plot([Iq_ref,Id_ref,Iexc_ref]);
legend('Iq','Id','Iexc');
subplot(3,1,2);
plot(CAN_VES_M_HM1_Ist'/94,'b'); 
hold on
plot(Mref,'r'); 
title('M zu Mref berechnet aus I'); 
xlabel('Messpunkte'); 
ylabel('M'); 
legend('M HM1','Mref');
subplot(3,1,3);
plot([CAN_VES_M_HM1_Ist'/94-Mref],'g');
title('M - Mref (Differenz)'); 
xlabel('Messpunkte'); 
ylabel('M-Mref'); 
legend('Differenz');

figure(2)
plot([M_Mittelwert',Mref_Mittelwert']);
legend('M HM1','Mref');


%% vergeleich zwischen Iq und Iq vom NeuronalenNetz Mittelwert statisch

Iq_NN_Werte = NeuralNetworkFunction_nn3_Mittelwert([Id_Mittelwert;If_Mittelwert;M_Mittelwert]);
figure(3)
plot([Iq_NN_Werte',Iq_Mittelwert']);
legend('Iq NeuronalNetwork statisch','Iq Mittelwert');

%% Neuronales Network lernen ohne If=0; (343 Werte)

index_if_0 = find(Eingang_nn3_Mittelwert(2,:)==0)
Eingang_nn3_Mittelwert(:,index_if_0) = []; %294 Werte
Ausgang_nn3_Mittelwert(:,index_if_0) = []; %294 Werte
Iq_Mittelwert(:,index_if_0) = []; %294 Werte

%Iq_ref = Iq_ref';
%Iq_ref(:,index_if_0) = [];
%% Neuronales Network lernen ohne Id~=0; (343 Werte)

index_id_0 = find(Eingang_nn3_Mittelwert(1,:)<=0.001 & Eingang_nn3_Mittelwert(1,:)>=-0.001)
Eingang_nn3_Mittelwert(:,index_id_0) = []; %315 Werte
Ausgang_nn3_Mittelwert(:,index_id_0) = []; %315 Werte
Iq_Mittelwert(:,index_id_0) = []; %315 Werte

%Iq_ref(:,index_id_0) = [];
%% Neuronales Network lernen ohne M~=0; (343 Werte)

index_m_0 = find(Eingang_nn3_Mittelwert(3,:)<=0.1)
Eingang_nn3_Mittelwert(:,index_m_0) = []; %247 Werte
Ausgang_nn3_Mittelwert(:,index_m_0) = []; %247 Werte
Iq_Mittelwert(:,index_m_0) = []; %247 Werte

%Iq_ref(:,index_m_0) = [];

%% vergleich zwischen Iq und Iq vom NeuronalenNetz Mittelwert (343) dynamisch
[iq_werte_Mittelwert,net_Mittelwert] = nn_manuell(17,Eingang_nn3_Mittelwert,Ausgang_nn3_Mittelwert);
figure(4)
subplot(2,1,1);

plot([iq_werte_Mittelwert',Iq_Mittelwert']);
hold on
legend('Iq NeuronalNetwork','Iq Mittelwert');

%juan
iq_from_nn_Juan = sim(net_Mittelwert,Eingang_nn3_Mittelwert)';
M_from_nn_Juan = Eingang_nn3_Mittelwert(3,:)';

iq_diff = (Iq_Mittelwert-iq_werte_Mittelwert)';
subplot(2,1,2)
plot(iq_diff);

%% erstellt eine Tabelle für die 10 grösten abweichungen von iq und iq_nn (Mittelwert)
max_diff_value=[]; 
for (i=1 : 10)
[M, A]=max(iq_diff ,[],1,'linear');
max_diff_value(i,:) = [Eingang_nn3_Mittelwert(:,A);Ausgang_nn3_Mittelwert(A);A]'% Id , If , M , Iq
iq_diff(A,:) = [];
end

%% Neuronales Network lernen ohne If=0; (5145 Werte)

Fehler_Eingangs_matrix = Eingang_nn3_alle;
Fehler_Ausgangs_matrix = Ausgang_nn3_alle;
index_if_0 = find(Eingang_nn3_alle(2,:)==0);
Fehler_Eingang_if = Fehler_Eingangs_matrix(:,index_id_0);
Fehler_Ausgang_if = Fehler_Ausgangs_matrix(:,index_id_0);
Fehler_if = [Eingang_nn3_alle(:,index_if_0);Ausgang_nn3_alle(:,index_if_0)];
Eingang_nn3_alle(:,index_if_0) = []; % Werte
Ausgang_nn3_alle(:,index_if_0) = []; % Werte
Iq_ref = Iq_ref';
Iq_ref(:,index_if_0) = []; %294 Werte

%% Neuronales Network lernen ohne Id~=0; (5145 Werte)

index_id_Fehler = find(Fehler_Eingangs_matrix(1,:)<=0.001 & Fehler_Eingangs_matrix(1,:)>=-0.001);
Fehler_id = [Fehler_Eingangs_matrix(:,index_id_Fehler);Fehler_Ausgangs_matrix(:,index_id_Fehler)];
Fehler_Eingang_id = Fehler_Eingangs_matrix(:,index_id_Fehler);
Fehler_Ausgang_id = Fehler_Ausgangs_matrix(:,index_id_Fehler);

index_id_0 = find(Eingang_nn3_alle(1,:)<=0.001 & Eingang_nn3_alle(1,:)>=-0.001);
%Fehler_id = [Eingang_nn3_alle(:,index_id_0);Ausgang_nn3_alle(:,index_id_0)];
Eingang_nn3_alle(:,index_id_0) = []; %315 Werte
Ausgang_nn3_alle(:,index_id_0) = []; %315 Werte
Iq_ref(:,index_id_0) = []; %315 Werte

%% Neuronales Network lernen ohne M~=0; (5145 Werte)

index_m_Fehler = find(Fehler_Eingangs_matrix(3,:)<=0.1);
Fehler_m = [Fehler_Eingangs_matrix(:,index_m_Fehler);Fehler_Ausgangs_matrix(:,index_m_Fehler)];
Fehler_Eingang_m = Fehler_Eingangs_matrix(:,index_m_Fehler);
Fehler_Ausgang_m = Fehler_Ausgangs_matrix(:,index_m_Fehler);

index_m_0 = find(Eingang_nn3_alle(3,:)<=0.1);
%Fehler_m = [Eingang_nn3_alle(:,index_m_0);Ausgang_nn3_alle(:,index_m_0)];
Eingang_nn3_alle(:,index_m_0) = []; %247 Werte
Ausgang_nn3_alle(:,index_m_0) = []; %247 Werte
Iq_ref(:,index_m_0) = []; %247 Werte

%% vergleich zwischen Iq und Iq vom NeuronalenNetz alle (51450) dynamisch
[iq_werte,net] = nn_manuell(17,Eingang_nn3_alle,Ausgang_nn3_alle);
%[sfsf,net_NN_1]=nn_manuell(17,[Ausgang_nn3_alle;Eingang_nn3_alle(1:2,:)],[Eingang_nn3_alle(3,:)]);
figure(5)
subplot(2,1,1);
plot((1:length(iq_werte)),iq_werte,(1:length(iq_werte)),Iq_ref);
legend('Iq NeuronalNetwork','Iq Mittelwert');

iq_diff = (iq_werte-Iq_ref)';
subplot(2,1,2)
plot(iq_diff);

%% plot fehler: Fehler_Id, Fehler_If, Fehler_m

figure(6)
subplot(3,1,1)
plot(Fehler_id');
legend('Id Strom','If Strom','M','Iq Strom')
subplot(3,1,2)
plot(Fehler_if');
legend('Id Strom','If Strom','M','Iq Strom')
subplot(3,1,3)
plot(Fehler_m');
legend('Id Strom','If Strom','M','Iq Strom')

%% Fehler mit NN nachbilden

%iq_werte_fehler = nn_manuell(17,[Fehler_Eingang_id,Fehler_Eingang_if,Fehler_Eingang_m],[Fehler_Ausgang_id,Fehler_Ausgang_if,Fehler_Ausgang_m]);
%figure(7)
%plot((1:length(iq_werte_fehler)),iq_werte_fehler,(1:length(iq_werte_fehler)),[Fehler_Ausgang_id,Fehler_Ausgang_if,Fehler_Ausgang_m])

%% optimization 343

global M_drehmoment_nn_3_Mittel;
%global iq_from_nn;
global i_Zaehlmarke;
M_drehmoment_nn_3_Mittel =(1:0.5:200);
%Id_from_nn = (1:1:198);
%If_from_nn = (1:0.1:20.7);
%input_nn_3 = ([Id_from_nn;If_from_nn;M_drehmoment_nn_3]); %;1,1,9.5
%iq_from_nn = sim(net_Mittelwert,input_nn_3)';
A = [];
b = [];
Aeq = [];
beq = [];

lb = [1 1];
ub = [400 400];
x0 = [1,1];

Rs = 7.1e-3;
Re = 7.26;
x_t_Mittel = [];

for(i_Zaehlmarke=1:length(M_drehmoment_nn_3_Mittel))
        %MFehler = @(x)M_drehmoment_nn_3(i_Zaehlmarke) - iq_from_nn(i_Zaehlmarke)*x(1)*x(2);
        %input_nn_3 = ([x(1);x(2);M_drehmoment_nn_3(i_Zaehlmarke)]); %;1,1,9.5
        %x(1) = sim(net_Mittelwert,[x(2);x(3);M_drehmoment_nn_3(i_Zaehlmarke)])';
PV =   @(x)(((((sim(net_Mittelwert,[x(1);x(2);M_drehmoment_nn_3_Mittel(i_Zaehlmarke)])')^2*Rs +  (x(1))^2*Rs)*(3/2)) +  (x(2))^2*Re));

        %options = optimoptions('fmincon','Display','iter','Algorithm','sqp');

        %nonlcon = @Nebenbedingung;
        %[x,fval,exitflag,output,lambda,grad,hessian] = fmincon(PV,x0,A,b,Aeq,beq,lb,ub,nonlcon,options);
x_Mittel = fmincon(PV,x0,A,b,Aeq,beq,lb,ub); % optimale Ströme  ,nonlcon
x_t_Mittel = [x_t_Mittel;[sim(net_Mittelwert,[x_Mittel M_drehmoment_nn_3_Mittel(i_Zaehlmarke)]') x_Mittel]];
i_Zaehlmarke
end
x=x_t_Mittel;

%% vergleich zwischen Iq und Iq vom NeuronalenNetz alle (343) dynamisch
[iq_werte_Mittel,net] = nn_manuell(17,Eingang_nn3_Mittelwert,Ausgang_nn3_Mittelwert);
%[sfsf,net_NN_1]=nn_manuell(17,[Ausgang_nn3_alle;Eingang_nn3_alle(1:2,:)],[Eingang_nn3_alle(3,:)]);
figure(5)
subplot(2,1,1);
plot((1:length(iq_werte_Mittel)),iq_werte_Mittel,(1:length(iq_werte_Mittel)),iq_werte_Mittelwert);
legend('Iq NeuronalNetwork Mittelwert','Iq Mittelwert');

iq_diff = (iq_werte_Mittel-iq_werte_Mittelwert)';
subplot(2,1,2)
plot(iq_diff);


%% optimization 5150

global M_drehmoment_nn_3;
%global iq_from_nn;
global i_Zaehlmarke;
M_drehmoment_nn_3 =(1:0.5:200.5);
%Id_from_nn = (1:1:198);
%If_from_nn = (1:0.1:20.7);
%input_nn_3 = ([Id_from_nn;If_from_nn;M_drehmoment_nn_3]); %;1,1,9.5
%iq_from_nn = sim(net_Mittelwert,input_nn_3)';
A = [];
b = [];
Aeq = [];
beq = [];

lb = [1 1];
ub = [200 100];
x0 = [1,1];

Rs = 7.1e-3;
Re = 7.26;
x_t = [];

%for(i_Zaehlmarke=1:length(M_drehmoment_nn_3))
%PV =   @(x)((((sim(net,[x(1);x(2);M_drehmoment_nn_3(i_Zaehlmarke)])')^2*Rs +  (x(1))^2*Rs)*(3/2) +  (x(2))^2*Re));
PV =   @(x)((((sim(net_Mittelwert,[x(1);x(2);M_drehmoment_nn_3(1)])')^2*Rs +  (x(1))^2*Rs)*(3/2) +  (x(2))^2*Re));

x = fmincon(PV,x0,A,b,Aeq,beq,lb,ub); % optimale Ströme  ,nonlcon
x_t = [x_t;[sim(net,[x M_drehmoment_nn_3(i_Zaehlmarke)]') x]];
%end
x=x_t;


%% Drehmoment mit Optimalen Iq, Id und If 343

%M_from_nn_1 = sim(net_NN_1,[iq_from_nn,2,1]')

M_drehmoment_nn_1_Mittel= myNeuralNetworkFunction_nn_1_ue_94([x_t_Mittel'])
diff_M_from_MSoll_to_NN_1= sqrt((M_drehmoment_nn_3_Mittel'-M_drehmoment_nn_1_Mittel').^2);
figure(7)
subplot(2,1,1)
plot((1:0.5:0.5+length(M_drehmoment_nn_3_Mittel)/2),M_drehmoment_nn_3_Mittel',(1:0.5:0.5+length(M_drehmoment_nn_3_Mittel)/2),M_drehmoment_nn_1_Mittel)
subplot(2,1,2)
plot((1:0.5:0.5+length(M_drehmoment_nn_3_Mittel)/2),diff_M_from_MSoll_to_NN_1)

figure(9)
plot(M_drehmoment_nn_3_Mittel,x_t_Mittel');
legend("Iq opt","Id opt","If opt")
[iq_werte_op,net_optimal] = nn_manuell(17,M_drehmoment_nn_3_Mittel,x_t_Mittel');%x-Achse muss noch halbiert betrachtet werden 400x0.5(Schrittweite)

save NN_optimal net_optimal;

%M_drehmoment_nn_1_2= myNeuralNetworkFunction_nn_1_ue_94([iq_from_nn,2,1]')
%diff_M_from_NN_3_to_NN_1= sqrt((M_drehmoment_nn_3-M_drehmoment_nn_1_2)^2)

%maxxx = max([Eingang_nn3_alle(3,:)]); Maximales Drehmoment (gelernt)