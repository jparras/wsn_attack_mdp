function [s_list,s_list_f,u_v,p_tr,reward,states_per_stage]= obtain_values(N,Ma,Mg,Mm,q,g,p1a,p1g,njam_max,u,disp_flag)

u_v = obtain_actions(Mm,njam_max,Ma);
% Initial state
r_g=zeros(1,2*(N+1)+1);
r_a=zeros(1,2*(N+1)+1);
r_g(N+2)=Mg;
r_a(N+2)=Ma;
s_list(1,:)=[r_g r_a njam_max];
states_actual=s_list;
states_next=[];
states_per_stage=cell(N+1,1);
states_per_stage{1}=states_actual;
% Step 1: obtain all possible states
for i=1:N
    if disp_flag==1
        display(['Obtaining states list: Stage ' num2str(i) ' of ' num2str(N) '; Number of states = ' num2str(size(s_list,1))]);
    end
    for is=1:size(states_actual,1) %For all states
        state=states_actual(is,:);
        r_g=state(1:2*(N+1)+1);
        r_a=state(2*(N+1)+2:end-1);
        njl=state(end);
        n_u=size(u_v{njl+1},1); %Number of possible actions in this state
        for ia=1:n_u %For all possible actions
            actions_a=u_v{njl+1}(ia,1:Ma);
            actions_g=u_v{njl+1}(ia,Ma+1:end);
            %Obtain all possible transition values
            [n_jam_left_out, rep_g_out, rep_m_out,pr_out,seq_out,p1m,p2m,p3m,p1v,p2v,p3v]=test_jam(Mm,q,g,r_g,r_a,p1a,p1g,actions_g,actions_a,-N-1:N+1,njl);
            % Update state values
            states_next=[states_next; rep_g_out rep_m_out n_jam_left_out]; % Update next state list
            states_next=unique(states_next,'rows'); %Take only the unique states
        end
    end
    states_actual=states_next;
    states_next=[];
    states_per_stage{i+1}=states_actual; %Update list of states
    if i<N %Do NOT include in s_list the FINAL STATES!!!
        s_list=[s_list; states_per_stage{i+1}]; % Update states list
        s_list=unique(s_list,'rows'); %Take only the unique states
    end
end
% Obtain state list with final states
s_list_f=[s_list; states_per_stage{i+1}]; % Update states list
s_list_f=unique(s_list_f,'rows'); %Take only the unique states
n_states=size(s_list,1);
n_states_f=size(s_list_f,1);
% %Figure plot
% s_norm=[ones(n_states_f,2*(N+1)+1)/Mg ones(n_states_f,2*(N+1)+1)/Ma];
% figure();
% imagesc(s_list_f.*s_norm);
% Step 2: Obtain all transition probabilities and rewards
n_u_max=0;
for i=1:size(u_v,1)
    if n_u_max<size(u_v{i},1)
        n_u_max=size(u_v{i},1);
    end
end
p_tr=zeros(n_states,n_u_max, n_states_f); %State_origin x action x State_destiny
reward=-Inf*ones(n_states,n_u_max, n_states_f); %State_origin x action x State_destiny
for s_orig=1:n_states
    if mod(s_orig,20)==0 && disp_flag==1
        display(['Obtaining rewards and transitions: State ' num2str(s_orig) ' of ' num2str(n_states)]);
    end
    state_orig=s_list(s_orig,:);
    r_g=state_orig(1:2*(N+1)+1);
    r_a=state_orig(2*(N+1)+2:end-1);
    njl=state_orig(end);
    n_u=size(u_v{njl+1},1); %Number of possible actions in this state
    for ia=1:n_u %For all possible actions
        actions_a=u_v{njl+1}(ia,1:Ma);
        actions_g=u_v{njl+1}(ia,Ma+1:end);
        %Obtain all possible transition values
        [n_jam_left_out, rep_g_out, rep_m_out,pr_out,seq_out,p1m,p2m,p3m,p1v,p2v,p3v]=test_jam(Mm,q,g,r_g,r_a,p1a,p1g,actions_g,actions_a,-N-1:N+1,njl);
        for s_dest=1:length(pr_out) %For all possible next states
            % Obtain transition probability
            next_state=[rep_g_out(s_dest,:) rep_m_out(s_dest,:) n_jam_left_out(s_dest,:)];
            [tf, index]=ismember(next_state,s_list_f,'rows');
            if tf==0
                error('State not found');
            end
            p_tr(s_orig,ia,index)=p_tr(s_orig,ia,index)+pr_out(s_dest);
            % Obtain reward
            if u==0
                reward(s_orig,ia,index)=p1v(s_dest)+p3v(s_dest);
            else
                reward(s_orig,ia,index)=p2v(s_dest);
            end
        end
    end
    % Check that probabilities are correctly set
    if abs(sum(sum(squeeze(p_tr(s_orig,:,:))'))-n_u)>1e-3 %To account for numerical errors
        error('Probabilities are incorreclty set');
    end
end