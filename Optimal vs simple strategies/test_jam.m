function [n_jam_left_out, rep_g_out, rep_m_out,pr_out,seq_out,p1,p2,p3,p1vo,p2vo,p3vo]=test_jam(Mm,q,g,rep_g,rep_m,p1m,p1g,actions_g,actions_a,rep_vector, n_jam_left);
if length(actions_a) ~= sum(rep_m) || length(actions_g)~=Mm
    error('Values of players dont fit');
end
p1=0;
p2=0;
p3=0;
p1v=zeros(1,4^Mm);
p2v=zeros(1,4^Mm);
p3v=zeros(1,4^Mm);
rep_g_out=zeros(4^Mm,length(rep_g));
rep_m_out=zeros(4^Mm,length(rep_m));
p_out=zeros(4^Mm,1);
n_jam_left_out=zeros(4^Mm,1);
seq_out=-ones(4^Mm,Mm);

% Obtain the weights
rep_v_g=zeros(1,sum(rep_g)); % Reputations vector (not histograms!)
aux=rep_g;
id=1;
for i=1:sum(rep_g)
    while aux(id)==0
        id=id+1;
    end
    rep_v_g(i)=rep_vector(id);
    aux(id)=aux(id)-1;
end
rep_v_m=zeros(1,sum(rep_m)); % Reputations vector (not histograms!)
aux=rep_m;
id=1;
for i=1:sum(rep_m)
    while aux(id)==0
        id=id+1;
    end
    rep_v_m(i)=rep_vector(id);
    aux(id)=aux(id)-1;
end

r_avg=(sum(rep_g.*rep_vector)+sum(rep_m.*rep_vector))/(sum(rep_g)+sum(rep_m));

weights=zeros(1,Mm); %Weight
%probs=zeros(1,Mm); %Probability of good node in the reputation
reps=sort([rep_v_g(:); rep_v_m(:)],1,'descend'); % Sorted reputations
reps=reps(1:Mm); 

for i=1:Mm
    weights(i)=(reps(i)+g)/(r_avg+g);
    if weights(i)<0
        weights(i)=0;
    end
%     num_g=sum(rep_v_g==reps(i));
%     num_m=sum(rep_v_m==reps(i));
    %probs(i)=num_g/(num_g+num_m);
end
il=0;

while il<=4^Mm-1
    njl=n_jam_left;
    ilaux=-1;
    jv=zeros(1,Mm); %Jammed vector
    seq=dec2base(il,4,Mm)-'0'; % 0: 1 good, 1: 0 good, 2: 1 malicious, 3: 0 malicious
    Wp=zeros(Mm,1);
    %Initialize p_seq
    seq_index=1;
    switch seq(seq_index)
        case 0 % 1 good
            ng=sum(rep_v_g==reps(seq_index));
            na=sum(rep_v_m==reps(seq_index));
            prb=ng/(ng+na);
            p_seq=log(p1g)+log(prb);
            if actions_g(1)==0 %Jam
                njl=njl-1;
                jv(seq_index)=1;
            end
            Wp(seq_index)=weights(seq_index);
        case 1 % 0 good
            ng=sum(rep_v_g==reps(seq_index));
            na=sum(rep_v_m==reps(seq_index));
            prb=ng/(ng+na);
            p_seq=log(1-p1g)+log(prb);
            if actions_g(1)==0 %Jam
                njl=njl-1;
                jv(seq_index)=1;
                Wp(seq_index)=weights(seq_index); %When jammed, consider that 1 is received!
            else
                Wp(seq_index)=-weights(seq_index);
            end
        case 2 % 1 malicious
            ng=sum(rep_v_g==reps(seq_index));
            na=sum(rep_v_m==reps(seq_index));
            prb=ng/(ng+na);
            if actions_a(1)==2 %The first AS attacks
                p_seq=log(p1m)+log(1-prb);
            else %The first AS does not attack
                p_seq=log(p1g)+log(1-prb);
            end
            Wp(seq_index)=weights(seq_index);
        case 3 % 0 malicious
            ng=sum(rep_v_g==reps(seq_index));
            na=sum(rep_v_m==reps(seq_index));
            prb=ng/(ng+na);
            if actions_a(1)==2 %The first AS attacks
                p_seq=log(1-p1m)+log(1-prb);
            else
                p_seq=log(1-p1g)+log(1-prb);
            end
            Wp(seq_index)=-weights(seq_index);
    end
    seq_index=2;
    while seq_index<=Mm
        ngr=sum(rep_v_g==reps(seq_index)); %Number of good sensors in reputation left to call
        nar=sum(rep_v_m==reps(seq_index)); %Number of malicious sensors in reputation left to call
        nan=0; %Number of previous AS
        ngn=0; % Number of previous good sensors
        for index=1:seq_index-1
            if seq(index)==2 || seq(index)==3 % AS called
                nan=nan+1;
            else
                ngn=ngn+1;
            end
            if reps(seq_index)==reps(index)
                if seq(index)==2 || seq(index)==3 % AS called before
                    nar=nar-1;
                elseif seq(index)==0 || seq(index)==1 % Good sensor called before
                    ngr=ngr-1;
                end
            end
        end
        
        switch seq(seq_index)
             case 0 % 1 good
                p_seq=p_seq+log(p1g)+log(ngr/(ngr+nar));
                if actions_g(ngn+1)==0 %Jam
                    njl=njl-1;
                    jv(seq_index)=1;
                end
                Wp(seq_index)=weights(seq_index);
            case 1 % 0 good
                p_seq=p_seq+log(1-p1g)+log(ngr/(ngr+nar));
                if actions_g(ngn+1)==0 %Jam
                    njl=njl-1;
                    jv(seq_index)=1;
                    Wp(seq_index)=weights(seq_index); %When jammed, consider that 1 is received!
                else
                    Wp(seq_index)=-weights(seq_index);
                end
            case 2 % 1 malicious
                if nan<length(actions_a) % Malicious sensors still can be called
                    if actions_a(nan+1)==2 %The AS attacks
                        p_seq=p_seq+log(p1m)+log(1-ngr/(ngr+nar));
                    else
                        p_seq=p_seq+log(p1g)+log(1-ngr/(ngr+nar));
                    end
                else
                    p_seq=p_seq-Inf;
                end
                Wp(seq_index)=weights(seq_index);
            case 3 % 0 malicious
                if nan<length(actions_a) 
                    if actions_a(nan+1)==2 % The AS attacks
                        p_seq=p_seq+log(1-p1m)+log(1-ngr/(ngr+nar));
                    else
                        p_seq=p_seq+log(1-p1g)+log(1-ngr/(ngr+nar));
                    end
                else
                    p_seq=p_seq-Inf;
                end
                Wp(seq_index)=-weights(seq_index);
        end
        if sum(Wp)>=q || sum(Wp)<=-q || seq_index==Mm % Out!
            seq=seq(1:seq_index); % The part of the sequence that actually matters!
            if length(seq)<Mm 
                ndif=Mm-length(seq);
                ilaux=il+4^ndif;
            end
            % Update probabilities
            p_seq=exp(p_seq);
            p_out(il+1)=p_seq;
            % Update sequence
            seq_out(il+1,1:length(seq))=seq;
            % Update jamming left
            n_jam_left_out(il+1)=njl;
             % Update error and reputations
            rep_g_out(il+1,:)=rep_g; 
            rep_m_out(il+1,:)=rep_m;
            if sum(Wp)>=q % Decision == 1
                p1=p1+p_seq;
                p1v(il+1)=1;
                p2v(il+1)=0;
                p3v(il+1)=0;
                for idx=1:length(seq)
                    dix=find(rep_vector==reps(idx));
                    if seq(idx)==0 || jv(idx)==1 % 1 good or jammed vector
                            rep_g_out(il+1,dix+1)=rep_g_out(il+1,dix+1)+1;
                            rep_g_out(il+1,dix)=rep_g_out(il+1,dix)-1;
                    elseif seq(idx)==1 %0 good
                            rep_g_out(il+1,dix-1)=rep_g_out(il+1,dix-1)+1;
                            rep_g_out(il+1,dix)=rep_g_out(il+1,dix)-1;
                    elseif seq(idx)== 2 % 1 malicious
                            rep_m_out(il+1,dix+1)=rep_m_out(il+1,dix+1)+1;
                            rep_m_out(il+1,dix)=rep_m_out(il+1,dix)-1;
                    elseif seq(idx)== 3 % 0 malicious
                            rep_m_out(il+1,dix-1)=rep_m_out(il+1,dix-1)+1;
                            rep_m_out(il+1,dix)=rep_m_out(il+1,dix)-1;
                    end
                end
            elseif sum(Wp)<=-q % Decision == 0
                p2=p2+p_seq;
                p1v(il+1)=0;
                p2v(il+1)=1;
                p3v(il+1)=0;
                for idx=1:length(seq)
                    dix=find(rep_vector==reps(idx));
                    if seq(idx)== 0 || jv(idx)==1% 1 good
                            rep_g_out(il+1,dix-1)=rep_g_out(il+1,dix-1)+1;
                            rep_g_out(il+1,dix)=rep_g_out(il+1,dix)-1;
                    elseif seq(idx) ==1 % 0 good
                            rep_g_out(il+1,dix+1)=rep_g_out(il+1,dix+1)+1;
                            rep_g_out(il+1,dix)=rep_g_out(il+1,dix)-1;
                    elseif seq(idx) ==2 % 1 malicious
                            rep_m_out(il+1,dix-1)=rep_m_out(il+1,dix-1)+1;
                            rep_m_out(il+1,dix)=rep_m_out(il+1,dix)-1;
                    elseif seq(idx) ==3 % 0 malicious
                            rep_m_out(il+1,dix+1)=rep_m_out(il+1,dix+1)+1;
                            rep_m_out(il+1,dix)=rep_m_out(il+1,dix)-1;
                    end
                end
            elseif seq_index==Mm
                p3=p3+p_seq;
                p1v(il+1)=0;
                p2v(il+1)=0;
                p3v(il+1)=1;
                for idx=1:length(seq)
                    dix=find(rep_vector==reps(idx));
                    if seq(idx) ==0 || jv(idx)==1 % 1 good
                            rep_g_out(il+1,dix+1)=rep_g_out(il+1,dix+1)+1;
                            rep_g_out(il+1,dix)=rep_g_out(il+1,dix)-1;
                    elseif seq(idx) == 1 % 0 good
                            rep_g_out(il+1,dix-1)=rep_g_out(il+1,dix-1)+1;
                            rep_g_out(il+1,dix)=rep_g_out(il+1,dix)-1;
                    elseif seq(idx) == 2 % 1 malicious
                            rep_m_out(il+1,dix+1)=rep_m_out(il+1,dix+1)+1;
                            rep_m_out(il+1,dix)=rep_m_out(il+1,dix)-1;
                    elseif seq(idx) == 3 % 0 malicious
                            rep_m_out(il+1,dix-1)=rep_m_out(il+1,dix-1)+1;
                            rep_m_out(il+1,dix)=rep_m_out(il+1,dix)-1;
                    end
                end
            end
            seq_index=Mm+1; %To exit the while
        else
            seq_index=seq_index+1;
        end
    end
    % Update il
    if ilaux==-1
        il=il+1;
    else
        il=ilaux;
    end
end


% Erase p=0 values
for i=4^Mm:-1:1
    if p_out(i)==0
        p_out(i)=[];
        rep_m_out(i,:)=[];
        rep_g_out(i,:)=[];
        seq_out(i,:)=[];
        n_jam_left_out(i)=[];
        p1v(i)=[];
        p2v(i)=[];
        p3v(i)=[];
    end
end

% % Merge values with same sequence values!
% [seq_out,ia,ic] = unique(seq_out,'rows'); %Obtain unique sequence values
% pr_out=zeros(max(ic),1);
% for i=1:length(ic)
%     pr_out(ic(i))=pr_out(ic(i))+p_out(i);
% end
% rep_g_out=rep_g_out(ia,:);
% rep_m_out=rep_m_out(ia,:);
% n_jam_left_out=n_jam_left_out(ia);
% p1v=p1v(ia);
% p2v=p2v(ia);
% p3v=p3v(ia);

%State aggregation!
states=[rep_g_out rep_m_out n_jam_left_out];
[states,ia,ic] = unique(states,'rows'); %Obtain unique states values
pr_out=zeros(max(ic),1);
p1vo=zeros(max(ic),1);
p2vo=zeros(max(ic),1);
p3vo=zeros(max(ic),1);
for i=1:length(ic)
    pr_out(ic(i))=pr_out(ic(i))+p_out(i);
end
for i=1:length(ic)
    p1vo(ic(i))=p1vo(ic(i))+p1v(i)*p_out(i)/pr_out(ic(i));
    p2vo(ic(i))=p2vo(ic(i))+p2v(i)*p_out(i)/pr_out(ic(i));
    p3vo(ic(i))=p3vo(ic(i))+p3v(i)*p_out(i)/pr_out(ic(i));
end
rep_g_out=states(:,1:length(rep_g));
rep_m_out=states(:,1+length(rep_g):end-1);
n_jam_left_out=states(:,end);
% p1v=p1v(ia);
% p2v=p2v(ia);
% p3v=p3v(ia);