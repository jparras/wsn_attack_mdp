function[pe1,pe2,pe3,pe_w]=EWSZOT_at_jam(Nmax,pe,u,q,n_iter,M,g,evil_th,pa,njam)

pe1=0;
pe2=0;
pe3=0;

evil_nodes=datasample(1:M,evil_th,'Replace',false); %Evil nodes

reputations=zeros(1,M);
weights=zeros(1,M);

pe_w=zeros(3,n_iter);

for it=1:n_iter
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

    while dec==0 %While decision is not taken
        node=node+1;
        er=binornd(1,pe,1,1); %Node commits an error measuring
        at=binornd(1,pa,1,1); % Attack probability
        % Generate decision by node
        if sum(sorted_nodes(node)==evil_nodes)==0 && njam>0 && evil_th>0 %Good node and jamming
             njam=njam-1;
             ui=1; %Conservative decission
        elseif sum(sorted_nodes(node)==evil_nodes)==0  %Good node and no jamming
            if er==0  %% Good node, no error
                ui=u;
            elseif er==1  %% Good node, error
                ui=double(not(u));
            end
        else %Malicious node
            if at==0 %No attack
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
        end
        uiv(end+1)=ui;
        % Update W
        w=weights(sorted_nodes(node));
        W=W+(-1)^(ui+1)*w;
        % Decide
        if W>=q
            u_dec=1;
            dec=1;
            pe1=pe1+1/n_iter;
            pe_w(1,it)=1;
        elseif W<=-q
            u_dec=0;
            dec=1;
            pe2=pe2+1/n_iter;
            pe_w(2,it)=1;
        elseif node==Nmax
            u_dec=1;
            dec=2;
            pe3=pe3+1/n_iter;
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