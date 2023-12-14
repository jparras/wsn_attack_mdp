function [Q,max_reward,learning_error,policy_QL]=Q_learning_solve(s_list_f,u_v,N,Mg,Ma,nepochs,a_init,a_min,a_dec,e_init,e_min,e_dec,alfa,Mm,Pc,u,q,g,njam_max,states_per_stage,disp_flag)

% Initialize Q(s,a) to 0
n_u_max=0;
for i=1:size(u_v,1)
    if n_u_max<size(u_v{i},1)
        n_u_max=size(u_v{i},1);
    end
end
% Obtain actions masks: they determine which actions are available in each
% stage
action_mask=zeros(njam_max+1,n_u_max);
for i=1:size(u_v,1)
    action_mask(i,1:size(u_v{i},1))=ones(1,size(u_v{i},1));
end
action_mask=logical(action_mask);

Q=zeros(size(s_list_f,1),n_u_max);
% Find initial state
r_g=zeros(1,2*(N+1)+1);
r_a=zeros(1,2*(N+1)+1);
r_g(N+2)=Mg;
r_a(N+2)=Ma;
[tf, initial_idx]=ismember([r_g r_a njam_max],s_list_f,'rows'); 
% Learn loop
learning_error=zeros(1,nepochs);
a=a_init/a_dec;
ep=e_init/e_dec;
for t=1:nepochs
    % Set a and ep
    if a>a_min
        a=a*a_dec;
    else 
        a=a_min;
    end
    if ep>e_min
        ep=ep*e_dec;
    else 
        ep=e_min;
    end
    if mod(t,1000)==0 && disp_flag==1
        display(['Q-learning: epoch = ' num2str(t) ' of ' num2str(nepochs)]);
    end
    s_idx=initial_idx; %Always start form the same state!
    for k=1:N % Each episode has exactly N steps!!
        % Work only with possible Q values
        Q_val=Q(s_idx,:);
        njam=s_list_f(s_idx,end); 
        n_u=sum(double(action_mask(njam+1,:))); %Number of possible actions in this stage
        Q_val=Q_val(action_mask(njam+1,:));
        % Choose action
        [~, a_idx]=max(Q_val); % Greedy action (used to explote)
        random_val=binornd(1,ep);
        if random_val==1 % Explore
            a_idx=datasample(1:n_u,1); % Randomly select one action
        end
        % Take action and observe r and s
        actions_a=u_v{njam+1}(a_idx,1:Ma);
        actions_g=u_v{njam+1}(a_idx,Ma+1:end);
        [r,s_next]=transition_EWSZOT(Mm,Pc,u,q,g,s_list_f(s_idx,:),-N-1:N+1,actions_g,actions_a);
        r=r/N; %Mean per stage!!
        % Update Q function
        [tf, s_next_idx]=ismember(s_next,s_list_f,'rows'); 
        Q_next=Q(s_next_idx,:); %Obtain only valid Q_next values (as a function of the actions!!)
        njam_next=s_list_f(s_next_idx,end); 
        Q_next=Q_next(action_mask(njam_next+1,:));
        Q_next_max=max(Q_next); %Max next Q value
        Q(s_idx,a_idx)= Q(s_idx,a_idx)+a*(r+alfa*Q_next_max-Q(s_idx,a_idx));
        % Update values for next iteration
        s_idx=s_next_idx;
    end
    learning_error(t)=max(Q(initial_idx,:));
end
max_reward=max(Q(initial_idx,:));

%Obtain policy in cell structure
policy_QL=cell(N,1); 
for k=1:N
    for j=1:size(states_per_stage{k},1)
        [tf, id]=ismember(states_per_stage{k}(j,:),s_list_f,'rows'); %Index in poli_QL!!
        Q_val=Q(id,:);
        njam=s_list_f(id,end); 
        Q_val=Q_val(action_mask(njam+1,:));
        % Choose action
        [~, idx_max]=max(Q_val); % Greedy action (used to explote)
        policy_QL{k}(j,:)=u_v{njam+1}(idx_max,:);
    end
end