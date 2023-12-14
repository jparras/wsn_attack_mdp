function [r,s_next]=transition_EWSZOT(Mm,pe,u,q,g,current_state,rep_vector,actions_g,actions_a)

% Initialize values
pe1=0;
pe2=0;
pe3=0;
%Obtain reputation vectos (histograms)
rep_g=current_state(1:length(rep_vector));
rep_m=current_state(length(rep_vector)+1:end-1);
njam=current_state(end);

% Obtain mean reputation value
r_avg=sum(rep_g.*rep_vector+rep_m.*rep_vector)/sum(rep_g+rep_m);

% Obtain rep values per stage (already sorted)
r=length(rep_vector); %Max reputation
rep_stages=[];
while length(rep_stages)<Mm
    rg=rep_g(r);
    rm=rep_m(r);
    if rg+rm>0
        for i=1:rg+rm
            rep_stages(end+1)=rep_vector(r);
        end
    end
    r=r-1;
end
rep_stages=rep_stages(1:Mm);
%Make a decision
dec=0;
W=0;
u_dec=[];
uiv=[];
rep_g_aux=rep_g; % These are modified in the loop!
rep_m_aux=rep_m;
node=0; %Number of sensors called
nodes_called=[]; %Type of nodes called
while dec==0 %While decision is not taken
    % Call a sensor
    node=node+1;
    rep=rep_stages(node); %Reputation required
    rg=rep_g_aux(rep_vector==rep); 
    rm=rep_m_aux(rep_vector==rep);
    node_idx_selected=datasample(1:rg+rm,1); %Evil nodes
    if node_idx_selected<=rg %Good node selected
        node_selected=1;
        rep_g_aux(rep_vector==rep)=rep_g_aux(rep_vector==rep)-1; %Erase this sensor for future calls
    else
        node_selected=-1;
        rep_m_aux(rep_vector==rep)=rep_m_aux(rep_vector==rep)-1; %Erase this sensor for future calls
    end
    nodes_called(end+1)=node_selected;
    % Error probability
    er=binornd(1,pe,1,1); %Node commits an error measuring
    % Generate decision by sensor
    if node_selected==1  %Good node
        if actions_g(1)==1 % No jamming
            if er==0  %% Good node, no error
                ui=u;
            elseif er==1  %% Good node, error
                ui=double(not(u));
            end
        else %Jamming
            ui=1;
            njam=njam-1;
        end
        actions_g(1)=[];
    else % Attacking sensor
        if actions_a(1)==3 %No attack
            if er==0  %% no error
                ui=u;
            elseif er==1  %% error
                ui=double(not(u));
            end
        else %Attack
            if er==0  %%  no error
                ui=double(not(u));
            elseif er==1  %% error
                ui=u;
            end
        end
        actions_a(1)=[]; % Erase the first value, so that the next AS uses the second one
    end
    uiv(end+1)=ui;
    % Update W
    w=(rep+g)/(r_avg+g);
    if w<0
        w=0;
    end
    W=W+(-1)^(ui+1)*w;
    % Decide
    if W>=q
        u_dec=1;
        dec=1;
        pe1=1;
    elseif W<=-q
        u_dec=0;
        dec=1;
        pe2=1;
    elseif node==Mm
        u_dec=1;
        dec=2;
        pe3=1;
    end
end
%Update reputations
for i=1:length(uiv)
    if u_dec==uiv(i) % Reputation increase
        rep=rep_stages(i)+1;
        if nodes_called(i)==1 %Good node
            rep_g_aux(rep_vector==rep)=rep_g_aux(rep_vector==rep)+1;
        else
            rep_m_aux(rep_vector==rep)=rep_m_aux(rep_vector==rep)+1;
        end
    else %Repuation decrease!
        rep=rep_stages(i)-1;
        if nodes_called(i)==1 %Good node
            rep_g_aux(rep_vector==rep)=rep_g_aux(rep_vector==rep)+1;
        else
            rep_m_aux(rep_vector==rep)=rep_m_aux(rep_vector==rep)+1;
        end
    end
end

% Output values
if u==0
    r=pe1+pe3;
else
    r=pe2;
end
s_next=[rep_g_aux rep_m_aux njam];