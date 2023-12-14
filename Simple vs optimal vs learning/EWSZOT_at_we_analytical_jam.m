function[Pe1,Pe2,Pe3,p_error]=EWSZOT_at_we_analytical_jam(Nmax,pe,q,nit,M,g,evil_th,pa,limit,njam);

Pe1=0;
Pe2=0;
Pe3=0;

if evil_th==0 % If there are no malicious node, then there is no jamming! 
    njam=0;
end

rep_vector=-nit:nit;
data_cont_old=zeros(limit,2*(2*nit+1)+2); % 1: prob, 2: njam_spent, 3:2+2*nit+3: rep_g, 2*nit+4:end: rep_m 

% Initialize values
rep_g=zeros(1,2*nit+1);
rep_m=zeros(1,2*nit+1);
rep_g(nit+1)=M-evil_th;
rep_m(nit+1)=evil_th;
data_cont_old(1,1)=1; %all probability to this initial node
data_cont_old(1,2)=0; % No jamming at the moment
data_cont_old(1,3:2*nit+3)=rep_g;
data_cont_old(1,2*nit+4:end)=rep_m;

p_error=zeros(1,nit);

for it=1:nit
%     display(['Iteration ' num2str(it) ' of ' num2str(nit)]);
    P1=0;
    P2=0;
    P3=0;
    data_cont_new=[];
    data_cont_old(end+1,:)=zeros(1,2*(2*nit+1)+2);
    n_iter=find(data_cont_old(:,1)==0,1)-1;
    for j=1:n_iter
        if data_cont_old(j,1)>0 %There is positive probability
            rep_g=data_cont_old(j,3:2*nit+3);
            rep_m=data_cont_old(j,2*nit+4:end);
            n_jam_left=njam-data_cont_old(j,2);
            pr=data_cont_old(j,1);
            p1g=pe;
            p1m=pe*(1-pa)+pa*(1-pe);
            [n_jam_left_out, rep_g_out,rep_m_out,pr_out,~,p1,p2,p3]=test_jam_an(Nmax,q,g,rep_g,rep_m,p1m,p1g,rep_vector,n_jam_left);
            pr_out=pr*pr_out;
            if j==1
                data_cont_new=[pr_out njam-n_jam_left_out rep_g_out rep_m_out];
            else
                data_cont_new=[data_cont_new; pr_out njam-n_jam_left_out rep_g_out rep_m_out];
            end
            % Merge together equal states
            [~,~,ic] = unique(data_cont_new(:,2:end),'rows'); %Obtain unique sequence values
            dcn=zeros(max(ic),size(data_cont_new,2));
            for i=1:length(ic)
                dcn(ic(i),1)=dcn(ic(i,1))+data_cont_new(i,1);
                dcn(ic(i),2:end)=data_cont_new(i,2:end);
            end
            data_cont_new=dcn;
            % Sort using probability and erase if there are too many values
            [~,idx] = sort(data_cont_new(:,1),'descend'); % sort just the first column
            data_cont_new = data_cont_new(idx,:); 
            if size(data_cont_new,1)>limit
                data_cont_new=data_cont_new(1:limit,:);
            end
            % Update errors
            P1=P1+p1*pr;
            P2=P2+p2*pr;
            P3=P3+p3*pr;
        end
    end
    % Error normalization
    psum=P1+P2+P3;
    p_error(it)=1-psum;
    P1=P1/psum;
    P2=P2/psum;
    P3=P3/psum;
    %Update data for next iteration
    data_cont_old=data_cont_new;
    % Update errors
    Pe1=Pe1+P1/nit;
    Pe2=Pe2+P2/nit;
    Pe3=Pe3+P3/nit;
end