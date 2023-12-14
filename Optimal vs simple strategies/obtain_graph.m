function [G]=obtain_graph(states_per_stage,s_list,s_list_f,u_v,policy,p_tr,reward,N,njam_max,Ma);

id=1;
states_number=cell(N+1,1);

for k=1:N+1
    states_number{k}=id:id+size(states_per_stage{k},1)-1;
    id=states_number{k}(end)+1;
end
names=cell(size(s_list_f,1),1);
id=1;
for k=1:N+1
    for j=1:size(states_per_stage{k},1)
        state=states_per_stage{k}(j,:);
        r_g=state(1:2*(N+1)+1);
        r_a=state(2*(N+1)+2:end-1);
        njl=state(end);
        aux='rg:';
        rv=-N-1:N+1;
        for i=1:length(r_g)
            if r_g(i)>0
                aux=[aux ' (' num2str(rv(i)) ')-' num2str(r_g(i)) ';'];
            end
        end
        aux=[aux ' ra:'];
        for i=1:length(r_a)
            if r_a(i)>0
                aux=[aux ' (' num2str(rv(i)) ')-' num2str(r_a(i)) ';'];
            end
        end
        aux=[aux ' ((' num2str(njam_max-njl) ')) ;'];
        if k<=N %Add optimal policy and unique ID
            a_a=policy{k}(j,1:Ma);
            a_g=policy{k}(j,Ma+1:end);
            a_a(a_a==2)=1;
            a_a(a_a==3)=0;
            a_g=double(not(a_g));
            aux=[aux ' a_a = ' mat2str(a_a) ', a_g = '  mat2str(a_g) ';id=' num2str(id)];
        end
        names{id}=aux;
        id=id+1;
    end
end
s=[]; % Source nodes
t=[]; % Target nodes
w=[]; % Weight of the edges
tp=[];
for k=1:N
    taux=[];
    states_s=states_per_stage{k};
    states_t=states_per_stage{k+1};
    for is=1:size(states_s,1)
        st_s=states_s(is,:); %Source state
        for it=1:size(states_t,1)
            st_t=states_t(it,:); %Target state
            njl=st_s(end); %njl in source state
            % Obtain transition probability given the optimal policy
            actions=policy{k}(is,:);
            [tfs, ids]=ismember(st_s,s_list,'rows');
            [tft, idt]=ismember(st_t,s_list_f,'rows');
            [tfa, ida]=ismember(actions,u_v{njl+1},'rows');
            if tfs==0 || tft==0 || tfa==0
                error('State was not found');
            end
            prob=p_tr(ids,ida,idt);
            if prob>0 
                if k==1 || (k>1 && sum(states_number{k}(is)==tp)>0)
                    s(end+1)=states_number{k}(is);
                    t(end+1)=states_number{k+1}(it);
                    w(end+1)=prob;
                    taux(end+1)=states_number{k+1}(it);
                    names{states_number{k+1}(it)}=[names{states_number{k+1}(it)} '; r =' num2str(reward(ids,ida,idt))];
                end
            end
        end
    end
    tp=taux;
end

G=digraph(s,t,w,names);
% Erase nodes that do not appear on the optimal path
G=subgraph(G,{names{unique([s,t])}});