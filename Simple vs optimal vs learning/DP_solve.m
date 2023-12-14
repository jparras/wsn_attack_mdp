function [optimal_reward,policy]=DP_solve(states_per_stage,s_list_f,s_list,p_tr,reward,u_v,N,alfa,disp_flag)

final_states=states_per_stage{N+1}; %Final states
% Final states reward
J_actual=zeros(1,size(final_states,1)); %Initialize to 0 the final reward

% DP algorithm (Bertsekas, Vol 2, sec 1.1.1, or Bertsekas, Vol 1, sec 1.3)
J_actual=alfa^N*J_actual;
policy=cell(N,1);
ip=1;
for k=1:N
    if disp_flag==1
        display(['Running DP algorithm iteration ' num2str(k) ' of ' num2str(N)]);
    end
    policy{N-k+1}=[];
    s_prev=states_per_stage{N-k+1};
    n_states=size(s_prev,1);
    J_prev=-Inf*ones(1,n_states);
    for s_p_i=1:n_states %Check for all states valid!
        [tf, index]=ismember(s_prev(s_p_i,:),s_list,'rows'); %Index in s_list (without final states)
        if tf==0
            error('State not found');
        end
        njl=s_prev(s_p_i,end);
        n_u=size(u_v{njl+1},1); %Number of possible actions in this state
        val=-Inf*ones(1,n_u);
        for ia=1:n_u % For all actions
            next_states_pr=squeeze(p_tr(index,ia,:));
            rew=squeeze(reward(index,ia,:));
            if abs(sum(next_states_pr)-1)>1e-3 %To account for numerical errors
                error('Probabilities are incorreclty set');
            end
            present_reward=nansum(next_states_pr.*rew)/N; % Expected reward!!!
            future_reward=0;
            for ipr=1:length(next_states_pr) %For all possible transitions
                if next_states_pr(ipr)>0 %There is positive probability
                    dest_state=s_list_f(ipr,:);
                    [tf2, idx2]=ismember(dest_state,states_per_stage{N-k+2},'rows');
                    if tf2==0
                        error('Destination state not found');
                    end 
                    future_reward=future_reward+next_states_pr(ipr)*J_actual(idx2);
                end
            end
            %future_reward=nansum(next_states_pr.*J_actual');
            val(ia)=alfa^(N-k)*present_reward+future_reward;
        end
        [J_prev(s_p_i), idm]=max(val);
        ip=ip+1;
        max_action=u_v{njl+1}(idm,:);
        policy{N-k+1}(end+1,:)=max_action;
    end
    J_actual=J_prev;
end

optimal_reward=J_actual;