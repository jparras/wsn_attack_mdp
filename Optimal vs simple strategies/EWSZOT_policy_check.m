function[pe1,pe2,pe3,pe_w]=EWSZOT_policy_check(Mm,pe,u,q,N,M,g,Ma,policy,states_per_stage,njam)

pe1=0;
pe2=0;
pe3=0;


evil_nodes=datasample(1:M,Ma,'Replace',false); %Evil nodes
good_nodes=setdiff(1:M,evil_nodes); %Good nodes

reputations=zeros(1,M);
weights=zeros(1,M);

pe_w=zeros(3,N);

for it=1:N
    dec=0;
    W=0;
    u_dec=[];
    [~,sorted_nodes]=sort(reputations,2,'descend'); % Sort
    %sorted_nodes=randi(M,[1,Nmax]); %Use a random assignation
    node=0;
    uiv=[];
    %Obtain weights
    weights(reputations>-g)=(reputations(reputations>-g)+g)/(mean(reputations)+g);
    weights(reputations<=-g)=0;
    if iscell(policy)==0
        if policy==1 %Always attack policy
            if njam>Mm
                optimal_policy_a=2*ones(1,Ma);
                optimal_policy_g=zeros(1,Mm);
            else
                optimal_policy_a=2*ones(1,Ma);
                optimal_policy_g=[zeros(1,njam) ones(1,Mm-njam)];
            end
        elseif policy==0 %Never attack policy
            optimal_policy_a=3*ones(1,Ma);
            optimal_policy_g=ones(1,Mm);
        end
    else %Optimal policy
        rep_v=-N-1:N+1;
        r_g=zeros(1,length(rep_v));
        r_a=zeros(1,length(rep_v));
        for i=1:length(r_g)
            r_g(i)=sum(reputations(good_nodes)==rep_v(i));
            r_a(i)=sum(reputations(evil_nodes)==rep_v(i));
        end
        state=[r_g r_a njam];
        [tf,idx]=ismember(state,states_per_stage{it},'rows');
        if tf==0
            error('State was not found');
        end
        optimal_policy_a=policy{it}(idx,1:Ma);
        optimal_policy_g=policy{it}(idx,Ma+1:end);
    end


    while dec==0 %While decision is not taken
        node=node+1;
        er=binornd(1,pe,1,1); %Node commits an error measuring
        % Generate decision by sensor
        if sum(sorted_nodes(node)==good_nodes)>0 %Good sensor 
            if optimal_policy_g(1)==1 % No jamming
                if er==0  %% Good node, no error
                    ui=u;
                elseif er==1  %% Good node, error
                    ui=double(not(u));
                end
            else %Jamming
                ui=1;
                njam=njam-1;
            end
            optimal_policy_g(1)=[];
        else %Malicious node
            if optimal_policy_a(1)==3 %No attack
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
            optimal_policy_a(1)=[]; % Erase the first value, so that the next AS uses the second one
        end
        uiv(end+1)=ui;
        % Update W
        w=weights(sorted_nodes(node));
        W=W+(-1)^(ui+1)*w;
        % Decide
        if W>=q
            u_dec=1;
            dec=1;
            pe1=pe1+1/N;
            pe_w(1,it)=1;
        elseif W<=-q
            u_dec=0;
            dec=1;
            pe2=pe2+1/N;
            pe_w(2,it)=1;
        elseif node==Mm
            u_dec=1;
            dec=2;
            pe3=pe3+1/N;
            pe_w(3,it)=1;
        end
    end
    %Update reputations
    for i=1:length(uiv)
        if u_dec==uiv(i)
            reputations(sorted_nodes(i))=reputations(sorted_nodes(i))+1;
        else
            reputations(sorted_nodes(i))=reputations(sorted_nodes(i))-1;
        end
    end
end