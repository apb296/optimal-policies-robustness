  close all
  

lambda0=0
mu0=1
s0=2
T=2000
obj=op

sample_path_lambda_tilde=ones(T,1)*lambda0;
            sample_path_mu=ones(T,1)*mu0;                        
            sample_path_s=ones(T,1)*s0;
            sample_path_pi=ones(T-1,1)*0;
            sample_path_V_a=ones(T-1,1)*0;
            sample_path_V_p=ones(T-1,1)*0;
            
            for t=2:T
                
                
                sample_path_s(t)=(discretesample(obj.PI(sample_path_s(t),:),1));
                sample_path_lambda_tilde(t)=obj.lambda_tilde{sample_path_s(t),sample_path_s(t-1)}([sample_path_mu(t-1),sample_path_lambda_tilde(t-1)]);
                sample_path_mu(t)=obj.mu{sample_path_s(t),sample_path_s(t-1)}([sample_path_mu(t-1),sample_path_lambda_tilde(t-1)]);
   %             sample_path_pi(t-1)=op.pi{sample_path_s(t-1)}([sample_path_mu(t-1),sample_path_lambda_tilde(t-1)] );                
   %             sample_path_V_p(t-1)=op.V_p{sample_path_s(t-1)}([sample_path_mu(t-1),sample_path_lambda_tilde(t-1)] );
   %             sample_path_V_a(t-1)=op.V_a{sample_path_s(t-1)}([sample_path_mu(t-1),sample_path_lambda_tilde(t-1)] );                
            end
           
            
            
    for s=1:obj.N
        index_s{s}=find(sample_path_s==s);
        sample_path_pi(index_s{s})=op.pi{s}([sample_path_mu(index_s{s}),sample_path_lambda_tilde(index_s{s})] );
        sample_path_V_a(index_s{s})=op.V_a{s}([sample_path_mu(index_s{s}),sample_path_lambda_tilde(index_s{s})] );
        sample_path_V_p(index_s{s})=op.V_p{s}([sample_path_mu(index_s{s}),sample_path_lambda_tilde(index_s{s})] );
        
    end
            
 
    figure()
    subplot(2,2,1)
    plot(sample_path_mu,'k','linewidth',1)
    xlabel('time')
    ylabel('$\mu$','interpreter','latex')
    
    subplot(2,2,2)
    plot(sample_path_lambda_tilde,'k','linewidth',1)
    xlabel('time')
    ylabel('$\tilde{\lambda}$','interpreter','latex')
    subplot(2,2,3)
    plot(sample_path_pi,'k','linewidth',1)
    xlabel('time')
    ylabel('$\pi$','interpreter','latex')
    subplot(2,2,4)
    plot(sample_path_V_a,'k','linewidth',1)
    hold on
    plot(sample_path_V_a,'r','linewidth',1)
    xlabel('time')
    ylabel('$V^i$','interpreter','latex')
%     
%     
% %      plot(sample_path_mu,sample_path_lambda_tilde)
%       figure()
%       subplot(2,1,1)
%             plot(sample_path_pi)
% 
%       
%       subplot(2,1,2)
%       plot(sample_path_mu,sample_path_lambda_tilde)
%       
%       figure()
%       plot(sample_path_mu)
%       
%       
%            figure()
%            
% 
%        plot(sample_path_V_a(1:end-1))
%        hold on
%        plot(sample_path_V_p(1:end-1),':r')
%                    