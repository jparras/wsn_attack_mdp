function [u_v] = obtain_actions(Mm,njam_max,Ma)
% ACTIONS CODE: 0: jam, 1: njam, 2: attack, 3: no attack
u_v=cell(njam_max+1,1);
% Possible attacking sensors actions
u_v_a=zeros(2^Ma,Ma); % Actions vector: number_of_actios x action
for i=1:2^Ma % Obtain all possible actions
    u_v_a(i,:)=dec2base(i-1,2,Ma)-'0';
end
u_v_a=u_v_a+2*ones(size(u_v_a)); % To match the coding!
% Possible good sensors actions
u_v_g=zeros(2^Mm,Mm); % Actions vector: number_of_actios x action
for i=1:2^Mm % Obtain all possible actions
    u_v_g(i,:)=dec2base(i-1,2,Mm)-'0';
end
for njam=0:njam_max
    u_v_aux=zeros(2^(Ma+Mm),Ma+Mm); %To store all possible actions
    id=1;
    for i=1:2^Mm % For all good node actions
        if sum(u_v_g(i,:)==0)<=njam
            u_v_aux(id:id+2^Ma-1,:)=[u_v_a repmat(u_v_g(i,:),2^Ma,1)]; %Store all good actions combinations
            id=id+2^Ma;
        end
    end
    u_v{njam+1}=u_v_aux(1:id-1,:);
end