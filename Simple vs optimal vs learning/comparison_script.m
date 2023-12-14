%% Comparison of AFA and JFA vs Optimal Bellman Strategies vs Q-learning
% Juan Parras, GAPS-UPM, December 2017
clear all; clc; close all;

%% Common parameters
nepochs=2*10^4;
a_init=0.5;
a_min=0.01;
a_dec=0.9997;
e_init=1;
e_min=0.01;
e_dec=0.9997;

nmW=6;

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
%% Case u=0
u=0;


error_u0_t_s=zeros(length(njamv),Np);
error_u0_t_o=zeros(length(njamv),Np);
error_u0_t_q=zeros(length(njamv),Np);

for i1=1:length(njamv)
    njam_max=njamv(i1);
    Pe1s=zeros(1,Np);
    Pe2s=zeros(1,Np);
    Pe3s=zeros(1,Np);
    Peo=zeros(1,Np);
    Peq=zeros(1,Np);
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
        % Optimal attack: obtain states
        [s_list,s_list_f,u_v,p_tr,reward,states_per_stage]= obtain_values(N,Ma,Mg,Mm,q,g,p1a,p1g,njam_max,u,0);
        % Optimal attack: DP solve
        [optimal_reward,policy]=DP_solve(states_per_stage,s_list_f,s_list,p_tr,reward,u_v,N,1,0);
        Peo(ip)=optimal_reward;
        % Q-learning:train
        [Q,max_reward,learning_error,policy_QL]=Q_learning_solve(s_list_f,u_v,N,Mg,Ma,nepochs,a_init,a_min,a_dec,e_init,e_min,e_dec,1,Mm,Pc,u,q,g,njam_max,states_per_stage,0);
        % Q-learning: check
        Pe1e=0;
        Pe2e=0;
        Pe3e=0;
        for rep=1:nrep
            [pe1t,pe2t,pe3t,~]=EWSZOT_policy_check(Mm,Pc,u,q,N,M,g,Ma,policy_QL,states_per_stage,njam_max);
            Pe2e=Pe2e+pe2t/nrep;
            Pe1e=Pe1e+pe1t/nrep;
            Pe3e=Pe3e+pe3t/nrep;
        end
        Peq(ip)=Pe1e+Pe3e;
    end
    error_u0_t_o(i1,:)=Peo;
    error_u0_t_s(i1,:)=Pe1s+Pe3s;
    error_u0_t_q(i1,:)=Peq;
end

%% Case u=1
u=1;


error_u1_t_s=zeros(length(njamv),Np);
error_u1_t_o=zeros(length(njamv),Np);
error_u1_t_q=zeros(length(njamv),Np);

for i1=1:length(njamv)
    njam_max=njamv(i1);
    Pe1s=zeros(1,Np);
    Pe2s=zeros(1,Np);
    Pe3s=zeros(1,Np);
    Peo=zeros(1,Np);
    Peq=zeros(1,Np);
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
        % Optimal attack: obtain states
        [s_list,s_list_f,u_v,p_tr,reward,states_per_stage]= obtain_values(N,Ma,Mg,Mm,q,g,p1a,p1g,njam_max,u,0);
        % Optimal attack: DP solve
        [optimal_reward,policy]=DP_solve(states_per_stage,s_list_f,s_list,p_tr,reward,u_v,N,1,0);
        Peo(ip)=optimal_reward;
        % Q-learning:train
        [Q,max_reward,learning_error,policy_QL]=Q_learning_solve(s_list_f,u_v,N,Mg,Ma,nepochs,a_init,a_min,a_dec,e_init,e_min,e_dec,1,Mm,Pc,u,q,g,njam_max,states_per_stage,0);
        % Q-learning: check
        Pe1e=0;
        Pe2e=0;
        Pe3e=0;
        for rep=1:nrep
            [pe1t,pe2t,pe3t,~]=EWSZOT_policy_check(Mm,Pc,u,q,N,M,g,Ma,policy_QL,states_per_stage,njam_max);
            Pe2e=Pe2e+pe2t/nrep;
            Pe1e=Pe1e+pe1t/nrep;
            Pe3e=Pe3e+pe3t/nrep;
        end
        Peq(ip)=Pe2e;
    end
    error_u1_t_o(i1,:)=Peo;
    error_u1_t_s(i1,:)=Pe2s;
    error_u1_t_q(i1,:)=Peq;
end

%% Add reference values 
u=0;
njam_max=0;
Ma=0;
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
end
error_u0_t_o(end+1,:)=Pe1th+Pe3th;
error_u0_t_s(end+1,:)=Pe1th+Pe3th;
error_u0_t_q(end+1,:)=Pe1th+Pe3th;

u=1;
njam_max=0;
Ma=0;
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
end
error_u1_t_o(end+1,:)=Pe2th;
error_u1_t_s(end+1,:)=Pe2th;
error_u1_t_q(end+1,:)=Pe2th;
%% Saving section

save('Data_comp2_final');
%% Plotting section
load('Data_comp2_final');
%load data from neural networks

error_u0_e_n=zeros(length(njamv),Np);
error_u1_e_n=zeros(length(njamv),Np);
error_u0_e_n_lstm=zeros(length(njamv),Np);
error_u1_e_n_lstm=zeros(length(njamv),Np);

% Obtain number of simulations
nsim=0;
d=dir([pwd filesep 'python_data']);
for i=1:length(d)
    if isempty(strfind(d(i).name,'DQN'))==0 && isempty(strfind(d(i).name,'mat'))==0
        aux_val=d(i).name;
        aux_val=aux_val(5:end);
        aux_val=aux_val(1:end-4);
        aux_val=str2double(aux_val);
        if aux_val>nsim
            nsim=aux_val;
        end
    end
end
% Load DQN case
for sim_index=1:nsim
    if exist([pwd filesep 'python_data' filesep 'DQN_' num2str(sim_index) '.mat'],'file')==2 % File exists
        load([pwd filesep 'python_data' filesep 'DQN_' num2str(sim_index) '.mat']);
        [~,id]=min(abs(piv-Pc));
        [~,cas]=min(abs(njamv-double(nj)));
        if u==0
            error_u0_e_n(cas,id)=reward_emp;
        elseif u==1
            error_u1_e_n(cas,id)=reward_emp;
        end
    end
end
% Load DRQN case
for sim_index=1:nsim
    if exist([pwd filesep 'python_data' filesep 'DRQN_' num2str(sim_index) '.mat'],'file')==2 % File exists
        load([pwd filesep 'python_data' filesep 'DRQN_' num2str(sim_index) '.mat']);
        [~,id]=min(abs(piv-Pc));
        [~,cas]=min(abs(njamv-double(nj)));
        if u==0
            error_u0_e_n_lstm(cas,id)=reward_emp;
        elseif u==1
            error_u1_e_n_lstm(cas,id)=reward_emp;
        end
    end
end
% Obtain plots
%u=0
for cas=1:3
    figure();
    plot(piv,error_u0_t_o(end,:),'g--','LineWidth',2);
    hold on;grid on;
    plot(piv,error_u0_t_o(cas,:),'b--','LineWidth',2);
    plot(piv,error_u0_t_s(cas,:),'r--','LineWidth',2);
    plot(piv,error_u0_t_q(cas,:),'k'); 
    plot(piv,error_u0_e_n(cas,:),'m'); 
    plot(piv,error_u0_e_n_lstm(cas,:),'m--'); 
    xlabel('P_c');
    ylabel('p_{e,t}');
end
% u=1
for cas=1:3
    figure();
    plot(piv,error_u1_t_o(end,:),'g--','LineWidth',2);
    hold on;grid on;
    plot(piv,error_u1_t_o(cas,:),'b--','LineWidth',2);
    plot(piv,error_u1_t_s(cas,:),'r--','LineWidth',2);
    plot(piv,error_u1_t_q(cas,:),'k'); 
    plot(piv,error_u1_e_n(cas,:),'m'); 
    plot(piv,error_u1_e_n_lstm(cas,:),'m--'); 
    xlabel('P_c');
    ylabel('p_{e,t}');
end

%% Poster comosens plots
figure();
cas=1;
subplot(1,2,1);
plot(piv,error_u0_t_o(end,:),'g','LineWidth',2);
hold on;grid on;
plot(piv,error_u0_t_o(cas,:),'b','LineWidth',2);
plot(piv,error_u0_t_s(cas,:),'r','LineWidth',2);
plot(piv,error_u0_e_n(cas,:),'k','LineWidth',2);
axis([0 0.5 0 0.8]);
xlabel('P_c');
ylabel('p_{e,t}');
title('SA');
cas=2;
subplot(1,2,2);
plot(piv,error_u0_t_o(end,:),'g','LineWidth',2);
hold on;grid on;
plot(piv,error_u0_t_o(cas,:),'b','LineWidth',2);
plot(piv,error_u0_t_s(cas,:),'r','LineWidth',2);
plot(piv,error_u0_e_n(cas,:),'k','LineWidth',2);
axis([0 0.5 0 0.8]);
xlabel('P_c');
ylabel('p_{e,t}');
title('CA');
set(gcf,'units','points','position',[10,10,1200,500])
print('fig_comosens_2018','-dpng')