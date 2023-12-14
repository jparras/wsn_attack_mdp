%% Comparison of AFA and JFA vs Optimal Bellman Strategies
% Juan Parras, GAPS-UPM, December 2017
clear all; clc; close all;

%% Case u=0
q=2;
Mm=4;
Np=51;
piv=linspace(0,1/2,Np);
g=5.51; 
nrep=100; %To average the empirical data

M=10;
Ma=1;
njamv=[0 1 2];
N=5;

limit=1e3; %Max number of th values stored
nmW=6;

u=0;


error_u0_t_s=zeros(length(njamv),Np);
error_u0_e_s=zeros(length(njamv),Np);
error_u0_t_o=zeros(length(njamv),Np);
error_u0_e_o=zeros(length(njamv),Np);

for i1=1:length(njamv)
    njam_max=njamv(i1);
    Pe1es=zeros(1,Np);
    Pe2es=zeros(1,Np);
    Pe3es=zeros(1,Np);
    Pe1s=zeros(1,Np);
    Pe2s=zeros(1,Np);
    Pe3s=zeros(1,Np);
    Peo=zeros(1,Np);
    Pe1eo=zeros(1,Np);
    Pe2eo=zeros(1,Np);
    Pe3eo=zeros(1,Np);
    parfor (ip=1:Np, nmW)
        display(['Iteration ' num2str(i1) ' of ' num2str(length(njamv)) ' ; ip = ' num2str(ip)]);
        Pc=piv(ip);
        Mg=M-Ma;
        if u==0
            p1g=Pc;
            p1a=1-Pc;
        else
            p1g=1-Pc;
            p1a=Pc;
        end
        % Simple attack: theoretical
        [Pe1s(ip),Pe2s(ip),Pe3s(ip),~]=EWSZOT_at_we_analytical_jam(Mm,Pc,q,N,M,g,Ma,1,limit,njam_max);
        % Simple attack: Empirical error
        for rep=1:nrep
            [pe1t,pe2t,pe3t]=EWSZOT_at_jam(Mm,Pc,u,q,N,M,g,Ma,1,njam_max);
            Pe2es(ip)=Pe2es(ip)+pe2t/nrep;
            Pe1es(ip)=Pe1es(ip)+pe1t/nrep;
            Pe3es(ip)=Pe3es(ip)+pe3t/nrep;
        end
        % Optimal attack: obtain states
        [s_list,s_list_f,u_v,p_tr,reward,states_per_stage]= obtain_values(N,Ma,Mg,Mm,q,g,p1a,p1g,njam_max,u,0);
        % Optimal attack: DP solve
        [optimal_reward,policy]=DP_solve(states_per_stage,s_list_f,s_list,p_tr,reward,u_v,N,1,0);
        Peo(ip)=optimal_reward;
        % Empirical error
        for rep=1:nrep
            [pe1t,pe2t,pe3t,~]=EWSZOT_policy_check(Mm,Pc,u,q,N,M,g,Ma,policy,states_per_stage,njam_max);
            Pe2eo(ip)=Pe2eo(ip)+pe2t/nrep;
            Pe1eo(ip)=Pe1eo(ip)+pe1t/nrep;
            Pe3eo(ip)=Pe3eo(ip)+pe3t/nrep;
        end
    end
    error_u0_t_o(i1,:)=Peo;
    error_u0_e_o(i1,:)=Pe1eo+Pe3eo;
    error_u0_t_s(i1,:)=Pe1s+Pe3s;
    error_u0_e_s(i1,:)=Pe1es+Pe3es;
end

figure();
plot(piv,error_u0_t_o,'b-', piv,error_u0_e_o,'b--', piv,error_u0_t_s,'r-', piv,error_u0_e_s,'r--'); grid on;
xlabel('P_c');
ylabel('p_{e,t}');

%% Case u=1
q=2;
Mm=4;
Np=51;
piv=linspace(0,1/2,Np);
g=5.51; 
nrep=100; %To average the empirical data

M=10;
Ma=1;
njamv=[0 1 2];
N=5;

limit=1e3; %Max number of th values stored
nmW=6;

u=1;


error_u1_t_s=zeros(length(njamv),Np);
error_u1_e_s=zeros(length(njamv),Np);
error_u1_t_o=zeros(length(njamv),Np);
error_u1_e_o=zeros(length(njamv),Np);

for i1=1:length(njamv)
    njam_max=njamv(i1);
    Pe1es=zeros(1,Np);
    Pe2es=zeros(1,Np);
    Pe3es=zeros(1,Np);
    Pe1s=zeros(1,Np);
    Pe2s=zeros(1,Np);
    Pe3s=zeros(1,Np);
    Peo=zeros(1,Np);
    Pe1eo=zeros(1,Np);
    Pe2eo=zeros(1,Np);
    Pe3eo=zeros(1,Np);
    parfor (ip=1:Np, nmW)
        display(['Iteration ' num2str(i1) ' of ' num2str(length(njamv)) ' ; ip = ' num2str(ip)]);
        Pc=piv(ip);
        Mg=M-Ma;
        if u==0
            p1g=Pc;
            p1a=1-Pc;
        else
            p1g=1-Pc;
            p1a=Pc;
        end
        % Simple attack: theoretical
        [Pe1s(ip),Pe2s(ip),Pe3s(ip),~]=EWSZOT_at_we_analytical_jam(Mm,1-Pc,q,N,M,g,Ma,1,limit,njam_max);
        % Simple attack: Empirical error
        for rep=1:nrep
            [pe1t,pe2t,pe3t]=EWSZOT_at_jam(Mm,Pc,u,q,N,M,g,Ma,1,njam_max);
            Pe2es(ip)=Pe2es(ip)+pe2t/nrep;
            Pe1es(ip)=Pe1es(ip)+pe1t/nrep;
            Pe3es(ip)=Pe3es(ip)+pe3t/nrep;
        end
        % Optimal attack: obtain states
        [s_list,s_list_f,u_v,p_tr,reward,states_per_stage]= obtain_values(N,Ma,Mg,Mm,q,g,p1a,p1g,njam_max,u,0);
        % Optimal attack: DP solve
        [optimal_reward,policy]=DP_solve(states_per_stage,s_list_f,s_list,p_tr,reward,u_v,N,1,0);
        Peo(ip)=optimal_reward;
        % Empirical error
        for rep=1:nrep
            [pe1t,pe2t,pe3t,~]=EWSZOT_policy_check(Mm,Pc,u,q,N,M,g,Ma,policy,states_per_stage,njam_max);
            Pe2eo(ip)=Pe2eo(ip)+pe2t/nrep;
            Pe1eo(ip)=Pe1eo(ip)+pe1t/nrep;
            Pe3eo(ip)=Pe3eo(ip)+pe3t/nrep;
        end
    end
    error_u1_t_o(i1,:)=Peo;
    error_u1_e_o(i1,:)=Pe2eo;
    error_u1_t_s(i1,:)=Pe2s;
    error_u1_e_s(i1,:)=Pe2es;
end

figure();
plot(piv,error_u1_t_o,'b-', piv,error_u1_e_o,'b--', piv,error_u1_t_s,'r-', piv,error_u1_e_s,'r--'); grid on;
xlabel('P_c');
ylabel('p_{e,t}');

%% Add reference values 
u=0;
njam_max=0;
Ma=0;
Pe1emp=zeros(1,Np);
Pe2emp=zeros(1,Np);
Pe3emp=zeros(1,Np);
Pe1th=zeros(1,Np);
Pe2th=zeros(1,Np);
Pe3th=zeros(1,Np);

parfor (ip=1:Np, nmW)
%for ip=1:Np
    display(['Iteration ' num2str(i1) ' of ' num2str(length(njamv)) ' ; ip = ' num2str(ip)]);
    Pc=piv(ip);
    Mg=M-Ma;
    if u==0
        p1g=Pc;
        p1a=1-Pc;
    else
        p1g=1-Pc;
        p1a=Pc;
    end
    % Simple attack: theoretical
    [Pe1th(ip),Pe2th(ip),Pe3th(ip),~]=EWSZOT_at_we_analytical_jam(Mm,Pc,q,N,M,g,Ma,1,limit,njam_max);
    % Simple attack: Empirical error
    for rep=1:nrep
        [pe1t,pe2t,pe3t]=EWSZOT_at_jam(Mm,Pc,u,q,N,M,g,Ma,1,njam_max);
        Pe2emp(ip)=Pe2emp(ip)+pe2t/nrep;
        Pe1emp(ip)=Pe1emp(ip)+pe1t/nrep;
        Pe3emp(ip)=Pe3emp(ip)+pe3t/nrep;
    end
end
error_u0_t_o(end+1,:)=Pe1th+Pe3th;
error_u0_e_o(end+1,:)=Pe1emp+Pe3emp;
error_u0_t_s(end+1,:)=Pe1th+Pe3th;
error_u0_e_s(end+1,:)=Pe1emp+Pe3emp;

%% Add reference values (u=0)
u=0;
njam_max=0;
Ma=0;
Pe1emp=zeros(1,Np);
Pe2emp=zeros(1,Np);
Pe3emp=zeros(1,Np);
Pe1th=zeros(1,Np);
Pe2th=zeros(1,Np);
Pe3th=zeros(1,Np);

parfor (ip=1:Np, nmW)
%for ip=1:Np
    display(['Iteration 1 of 2 ; ip = ' num2str(ip)]);
    Pc=piv(ip);
    Mg=M-Ma;
    if u==0
        p1g=Pc;
        p1a=1-Pc;
    else
        p1g=1-Pc;
        p1a=Pc;
    end
    % Simple attack: theoretical
    [Pe1th(ip),Pe2th(ip),Pe3th(ip),~]=EWSZOT_at_we_analytical_jam(Mm,Pc,q,N,M,g,Ma,1,limit,njam_max);
    % Simple attack: Empirical error
    for rep=1:nrep
        [pe1t,pe2t,pe3t]=EWSZOT_at_jam(Mm,Pc,u,q,N,M,g,Ma,1,njam_max);
        Pe2emp(ip)=Pe2emp(ip)+pe2t/nrep;
        Pe1emp(ip)=Pe1emp(ip)+pe1t/nrep;
        Pe3emp(ip)=Pe3emp(ip)+pe3t/nrep;
    end
end
error_u0_t_o(end+1,:)=Pe1th+Pe3th;
error_u0_e_o(end+1,:)=Pe1emp+Pe3emp;
error_u0_t_s(end+1,:)=Pe1th+Pe3th;
error_u0_e_s(end+1,:)=Pe1emp+Pe3emp;

u=1;
njam_max=0;
Ma=0;
Pe1emp=zeros(1,Np);
Pe2emp=zeros(1,Np);
Pe3emp=zeros(1,Np);
Pe1th=zeros(1,Np);
Pe2th=zeros(1,Np);
Pe3th=zeros(1,Np);

parfor (ip=1:Np, nmW)
%for ip=1:Np
    display(['Iteration 2 of 2 ; ip = ' num2str(ip)]);
    Pc=piv(ip);
    Mg=M-Ma;
    if u==0
        p1g=Pc;
        p1a=1-Pc;
    else
        p1g=1-Pc;
        p1a=Pc;
    end
    % Simple attack: theoretical
    [Pe1th(ip),Pe2th(ip),Pe3th(ip),~]=EWSZOT_at_we_analytical_jam(Mm,1-Pc,q,N,M,g,Ma,1,limit,njam_max);
    % Simple attack: Empirical error
    for rep=1:nrep
        [pe1t,pe2t,pe3t]=EWSZOT_at_jam(Mm,Pc,u,q,N,M,g,Ma,1,njam_max);
        Pe2emp(ip)=Pe2emp(ip)+pe2t/nrep;
        Pe1emp(ip)=Pe1emp(ip)+pe1t/nrep;
        Pe3emp(ip)=Pe3emp(ip)+pe3t/nrep;
    end
end
error_u1_t_o(end+1,:)=Pe2th;
error_u1_e_o(end+1,:)=Pe2emp;
error_u1_t_s(end+1,:)=Pe2th;
error_u1_e_s(end+1,:)=Pe2emp;

%% Obtain plots
%u=0
for cas=1:3
    figure();
    plot(piv,error_u0_t_o(end,:),'g:','LineWidth',2); 
    hold on;grid on;
    plot(piv,error_u0_e_o(end,:),'g-'); 
    plot(piv,error_u0_t_o(cas,:),'b:','LineWidth',2); 
    plot(piv,error_u0_e_o(cas,:),'b-'); 
    plot(piv,error_u0_t_s(cas,:),'r:','LineWidth',2); 
    plot(piv,error_u0_e_s(cas,:),'r-'); 
    xlabel('P_c');
    ylabel('p_{e,t}');
end
% u=1
for cas=1:3
    figure();
    plot(piv,error_u1_t_o(end,:),'g:','LineWidth',2); 
    hold on;grid on;
    plot(piv,error_u1_e_o(end,:),'g-'); 
    plot(piv,error_u1_t_o(cas,:),'b:','LineWidth',2); 
    plot(piv,error_u1_e_o(cas,:),'b-'); 
    plot(piv,error_u1_t_s(cas,:),'r:','LineWidth',2); 
    plot(piv,error_u1_e_s(cas,:),'r-'); 
    xlabel('P_c');
    ylabel('p_{e,t}');
end

%% SA strategy analysis
disp('Obtaining graphs...');
u=0;
njam_max=0;
Pc=0;
Ma=1;
Mg=M-Ma;
N=5;
if u==0
    p1g=Pc;
    p1a=1-Pc;
else
    p1g=1-Pc;
    p1a=Pc;
end
% Obtain states
[s_list,s_list_f,u_v,p_tr,reward,states_per_stage]= obtain_values(N,Ma,Mg,Mm,q,g,p1a,p1g,njam_max,u,0);
% Obtain optimal strategies
[optimal_reward,policy]=DP_solve(states_per_stage,s_list_f,s_list,p_tr,reward,u_v,N,1,0);
%Obtain graph
G=obtain_graph(states_per_stage,s_list,s_list_f,u_v,policy,p_tr,reward,N,njam_max,Ma);
%Plot graph
figure();
plot(G,'EdgeLabel',G.Edges.Weight);
title('Optimal graph');
% Obtain always attack graph
policy_aa=policy;
for k=1:N
    for i=1:size(policy_aa{k},1)
        policy_aa{k}(i,1:Ma)=2*ones(1,1:Ma);
    end
end
Gaa=obtain_graph(states_per_stage,s_list,s_list_f,u_v,policy_aa,p_tr,reward,N,njam_max,Ma);
figure();
plot(Gaa,'EdgeLabel',Gaa.Edges.Weight);
title('Optimal graph');

%% CLA strategy analysis
disp('Obtaining graphs...');
u=0;
njam_max=1;
Pc=0;
Ma=1;
Mg=M-Ma;
N=5;
if u==0
    p1g=Pc;
    p1a=1-Pc;
else
    p1g=1-Pc;
    p1a=Pc;
end
% Obtain states
[s_list,s_list_f,u_v,p_tr,reward,states_per_stage]= obtain_values(N,Ma,Mg,Mm,q,g,p1a,p1g,njam_max,u,0);
% Obtain optimal strategies
[optimal_reward,policy]=DP_solve(states_per_stage,s_list_f,s_list,p_tr,reward,u_v,N,1,0);
%Obtain graph
G=obtain_graph(states_per_stage,s_list,s_list_f,u_v,policy,p_tr,reward,N,njam_max,Ma);
%Plot graph
figure();
plot(G,'EdgeLabel',G.Edges.Weight);
title('Optimal graph');
% Obtain jfa graph
policy_jfa=policy;
for k=1:N
    for i=1:size(policy_jfa{k},1)
        njl=states_per_stage{k}(i,end);
        policy_jfa{k}(i,1:Ma)=2*ones(1,1:Ma);
        policy_jfa{k}(i,Ma+1:Ma+njl)=zeros(1,njl);
        policy_jfa{k}(i,Ma+njl+1:end)=ones(size(policy_jfa{k}(i,Ma+njl+1:end)));
    end
end
Gaa=obtain_graph(states_per_stage,s_list,s_list_f,u_v,policy_jfa,p_tr,reward,N,njam_max,Ma);
figure();
plot(Gaa,'EdgeLabel',Gaa.Edges.Weight);
title('Optimal graph');
%% Saving section

save('Data_comp_final');