% This file computes the optimal policy with robustness for a discrete
% state markov process for cost push shocks
classdef OPRob< handle
    
    properties
        
        
        %% Primitives
        
        % 1] Shocks
        N  % cardinality of state space
        Z  % State Space
        PI % Transition kernel
        
        % 2] Preferences and technology
        alpha_a_x % Agent's weight on output-gap fluctuations
        alpha_p_x % Planner's weight on output gap fluctuations
        
        alpha_a_pi % Agent's weight on inflation fluctuations
        alpha_p_pi % Planner's weight on inflation fluctuations
        
        
        beta    % subjective discount factor
        kappa   % degree of nominal frictions
        xstar % steady state output gap
        theta_a % robustness paramter for agent
        theta_p % robustness paramter for planner
        
        
        % Approximation
        
        % Grid for relative Pareto weights for Agent 2
        mu_min % lower bound
        mu_max % upper bound
        mu_grid % grid
        mu_grid_size % number of points in the grid
        mu_approx_order % order of approximation
        
        
        lambda_tilde_min % lower bound
        lambda_tilde_max% upper bound
        lambda_tilde_grid_size % number of points in the grid
        lambda_tilde_grid % grid
        lambda_tilde_approx_order % order of approximation
        
        domain % this will just combine the grid for costpush shock, mu and lambda_tilde
        grelax % this control the updating of policy rules recursions
        tol % error tolerance
        
        fspace % a collection of function spaces (one each for the discrete state)
        type_of_polynomial % basis polynomial
        
        
        
        % PolicyRules
        % 1] Inflation
        pi % function handle for policy rules for inflation
        coeff_pi % coeffs for the policy rules
        list_pi % values of inflation on the domain
        
        
        
        % 2] output gap
        x % function handle for policy rules for output gap
        coeff_x % coeffs for the output gap
        list_x % values of output on the entire domain
        
        
        % 3] committment multiplier
        lambda_tilde % function handle for policy rules for committment multiplier
        coeff_lambda_tilde % coeffs for the committment multiplier
        list_lambda_tilde % values of committment multiplier on the entire domain
        
        
        
        % 4] relative pareto weights
        mu % function handle for policy rules for relative pareto weights
        coeff_mu  % coeffs for the committment relative pareto weights
        list_mu  % values of relative pareto weights on the entire domain
        
        
        % 4] Agents values
        V_a % function handle for policy rules for Agents values
        coeff_V_a  % coeffs for the committment Agents values
        list_V_a  % Agents values on the entire domain
        list_m_p_star
        
        
        % 5] Planner's values
        V_p % function handle for policy rules for Planner's values
        coeff_V_p  % coeffs for the committment Planner's values
        list_V_p  % Planner's  values on the entire domain
        list_m_a_star
        
        list_error
        
        
        
    end
    
    
    methods
        
        function obj=OPRob(obj)
            addpath(genpath(pwd));            
            obj.N=2;
            obj.Z=(linspace(.9,1.1,obj.N)-1);
            obj.PI=ones(obj.N,obj.N)/obj.N;
            %obj.PI=[.6 .4;.4 .6];
            obj.alpha_a_x=1;
            obj.alpha_p_x=1;
            obj.alpha_a_pi=1;
            obj.alpha_p_pi=1;
            
            obj.beta=.98;
            obj.kappa=.5;
            obj.xstar=0;
            obj.mu_grid_size =10;
            obj.mu_approx_order=9;
            obj.type_of_polynomial='spli';
            obj.lambda_tilde_grid_size =10;
            obj.lambda_tilde_approx_order=9;
            obj.theta_a=1000000;
            obj.theta_p=1000000;
            
            
            obj.tol=1e-5;
            obj.mu_min=0;
            obj.mu_max=3;
            
            op_norob=OPNoRob(obj,obj.mu_min);
            
            op_norob.solveCrossEq();
            
            lambda_tilde_mu_min=op_norob.Gamma_lambda_0./(1-op_norob.Gamma_lambda_lambda);
            
            op_norob=OPNoRob(obj,obj.mu_max);
            
            op_norob.solveCrossEq();
            
            lambda_tilde_mu_max=op_norob.Gamma_lambda_0./(1-op_norob.Gamma_lambda_lambda);
            for s=1:obj.N
            obj.lambda_tilde_min(s)=min([lambda_tilde_mu_min;lambda_tilde_mu_max]);
            obj.lambda_tilde_max(s)=max([lambda_tilde_mu_min;lambda_tilde_mu_max]);
            end
            
            
        end
        
        function [value_pi, value_a, value_p]=solve_no_robustness_policies(obj,state)
            % This function uses the policy rules from the no robustness
            % case to compute inflation and value for a given point in the
            % domain. The argument state is a structure that store mu,
            % lambda_tilde and s
         
            op_norob=OPNoRob(obj,state.mu);
            op_norob.solveCrossEq();
            value_pi=op_norob.pi{state.s}(state.lambda_tilde);            
            value_x=(obj.kappa/(obj.alpha_p_x+state.mu*obj.alpha_a_x))*(state.lambda_tilde-value_pi*(obj.alpha_p_pi+state.mu*obj.alpha_a_pi))+obj.xstar;
            value_a=-(1/(1-obj.beta))*.5*(obj.alpha_a_pi*value_pi^2+obj.alpha_a_x*(value_x-obj.xstar)^2);
            value_p=-(1/(1-obj.beta))*.5*(obj.alpha_p_pi*value_pi^2+obj.alpha_p_x*(value_x-obj.xstar)^2);                                  
        end
        
        
        function build_grid(obj)
            % This function poulates the domain and sets up the functional space
            
            obj.domain=[];
            obj.fspace=[];
            obj.mu_grid=linspace(obj.mu_min,obj.mu_max, obj.mu_grid_size);
            for s=1:obj.N
                obj.lambda_tilde_grid(s,:)=linspace(obj.lambda_tilde_min(s),obj.lambda_tilde_max(s), obj.lambda_tilde_grid_size);                
               tempfspace= fundefn(obj.type_of_polynomial,[obj.mu_approx_order obj.lambda_tilde_approx_order], [obj.mu_min obj.lambda_tilde_min(s)],[obj.mu_max obj.lambda_tilde_max(s)]);         
               obj.fspace{s}=tempfspace;
            end
            d=1;
            for s=1:obj.N
                for mu_ind=1:obj.mu_grid_size
                    for lambda_tilde_ind=1:length(obj.lambda_tilde_grid(s,:))
                        obj.domain(d,:)=[obj.mu_grid(mu_ind), obj.lambda_tilde_grid(s,lambda_tilde_ind),s];
                        d=d+1;
                    end
                end
            end
            
        
            
        end
        
        function initialize(obj)
            for d =1:length(obj.domain)
                % extract the state
                state.mu=obj.domain(d,1);
                state.lambda_tilde=obj.domain(d,2);
                state.s=obj.domain(d,3);
                
                [value_pi, value_a, value_p]=obj.solve_no_robustness_policies(state);                
                obj.list_pi(d,1)=value_pi;
                obj.list_V_a(d,1)=value_a;
                obj.list_V_p(d,1)=value_p;
                obj.list_x(d,1)=(obj.kappa*(state.lambda_tilde-value_pi*(obj.alpha_p_pi+state.mu*obj.alpha_a_pi)))/(obj.alpha_p_x+obj.alpha_a_x*state.mu);
                for s_star=1:obj.N
                
                obj.list_m_p_star(d,s_star)=1;
                obj.list_m_a_star(d,s_star)=1;                
                obj.list_mu(d,s_star)=state.mu;
                obj.list_lambda_tilde(d,s_star)=(state.lambda_tilde-value_pi*(obj.alpha_p_pi+state.mu*obj.alpha_a_pi));
                
                end
                
            end
            
        end
        
        
        function [res, user, iflag]=res_non_linear_solver(obj,state,n,zstar,user,iflag)
            % This function solves for residual in the FOC that gets
            % mu_star, lambda_tilde_star and the implied distortions
            % m_p_star, m_a_star
            
            % Retrive the state variables today
            mu_today=state.mu;
            lambda_tilde_today=state.lambda_tilde;
            s=state.s;
            pi_today=state.pi;
            
            % Retrive the guess
            m_p_star=zstar(1:obj.N);
            m_a_star=zstar(obj.N+1:2*obj.N);
            lambda_tilde_star=zstar(2*obj.N+1:3*obj.N);
            mu_star=zstar(3*obj.N+1:4*obj.N);
%            pi_today=zstar(3*obj.N+1:4*obj.N+1:end);
            
            % Use the guessed functionals to compute values and inflation tomorrow
            for s_star=1:obj.N
                value_p_star(s_star)=obj.V_p{s_star}([mu_star(s_star) lambda_tilde_star(s_star)]);
                value_a_star(s_star)=obj.V_a{s_star}([mu_star(s_star) lambda_tilde_star(s_star)]);
                pi_star(s_star)=obj.pi{s_star}([mu_star(s_star) lambda_tilde_star(s_star)]);                                
            end
            
            % compute lambda from the foc with respect to pi
            lambda=(lambda_tilde_today-pi_today*(obj.alpha_p_pi+mu_today*obj.alpha_a_pi));
            
            
            % m_p_star
            Rhs_1=exp( -(1/(obj.theta_p)).*(value_p_star));                     
            ERhs_1=sum(obj.PI(s,:).*Rhs_1);
            Rhs_1=Rhs_1./ERhs_1;
            res(1,1:obj.N)=m_p_star-Rhs_1;
            
            
            % m_a_star
            Rhs_2=exp( -(1/(obj.theta_a)).*(value_a_star));
            ERhs_2=sum(obj.PI(s,:).*Rhs_2);
            Rhs_2=Rhs_2./ERhs_2;                                               
            res(1,obj.N+1:2*obj.N)=m_a_star-Rhs_2;
            
            
            
            
            % lambda_star
            res(1,2*obj.N+1:3*obj.N)=lambda_tilde_star-lambda*m_a_star./m_p_star;
            
            
            %mu_star
            res(1,3*obj.N+1:4*obj.N)=mu_star-(mu_today*m_a_star./m_p_star-pi_star.*(lambda*m_a_star./m_p_star).*(1-obj.PI(s,:).*m_a_star)/obj.theta_a );
            
            
            
 %           res(1,obj.N+1:2*obj.N+1)= pi_today-1/(pi_coeff_today)*(lambda_tilde_today*obj.kappa^2./(obj.alpha_p_x+mu_today*obj.alpha_a_x) +  obj.kappa*obj.xstar+obj.Z(s) + obj.beta*(sum(obj.PI(state.s,:).*m_p_star.*pi_star)));

            
            
        end
        
        
        function get_new_policies_on_grid(obj, ind_d)
        
            
            % initial guess 
            state.mu=obj.domain(ind_d,1);
            state.lambda_tilde=obj.domain(ind_d,2);
            state.s=obj.domain(ind_d,3);
           state.pi=obj.list_pi(ind_d,1);
            
            
            
            
            
            zstar0=[obj.list_m_p_star(ind_d,:) obj.list_m_a_star(ind_d,:) obj.list_lambda_tilde(ind_d,:) obj.list_mu(ind_d,:),obj.list_pi(ind_d,1)];
            
            
            funn=@(n,zstar,user,iflag) obj.res_non_linear_solver(state,n,zstar,user,iflag);
            [zstar, fvec, user, ifail] = nag_roots_sys_func_easy(funn, zstar0);
            
            if (ifail>0 && ind_d>1)
                disp('state')
                [obj.domain(ind_d,:)]
                            zstar0=[obj.list_m_p_star(ind_d-1,:) obj.list_m_a_star(ind_d-1,:) obj.list_lambda_tilde(ind_d-1,:) obj.list_mu(ind_d-1,:),obj.list_pi(ind_d-1,:)];
                [zstar, fvec, user, ifail] = nag_roots_sys_func_easy(funn, zstar0);
            disp(ifail)
            end
            
            % Retrive the solution
            obj.list_m_p_star(ind_d,:)=zstar(1:obj.N);
            obj.list_m_a_star(ind_d,:)=zstar(obj.N+1:2*obj.N);
            obj.list_lambda_tilde(ind_d,:)=zstar(2*obj.N+1:3*obj.N);
            obj.list_mu(ind_d,:)=zstar(3*obj.N+1:4*obj.N);
            %obj.list_pi(ind_d,1)=zstar(3*obj.N+1:4*obj.N+1:end);
            
            for s_star=1:obj.N
                value_p_star(s_star)=obj.V_p{s_star}([obj.list_mu(ind_d,s_star) obj.list_lambda_tilde(ind_d,s_star)]);
                value_a_star(s_star)=obj.V_a{s_star}([obj.list_mu(ind_d,s_star) obj.list_lambda_tilde(ind_d,s_star)]);
                pi_star(s_star)=obj.pi{s_star}([obj.list_mu(ind_d,s_star) obj.list_lambda_tilde(ind_d,s_star)]);                                
            end
            
            pi_coeff=(1+obj.kappa^2*(obj.alpha_p_pi+state.mu*obj.alpha_a_pi)./(obj.alpha_p_x+state.mu*obj.alpha_a_x));
        
            % check this algebra
           
            obj.list_pi(ind_d)= 1/(pi_coeff)*(state.lambda_tilde*obj.kappa^2./(obj.alpha_p_x+state.mu*obj.alpha_a_x) +  obj.kappa*obj.xstar+obj.Z(state.s) + obj.beta*(sum(obj.PI(state.s,:).*obj.list_m_p_star(ind_d,:).*pi_star)));
              
             obj.list_x(ind_d,1)=(obj.xstar+ obj.kappa*(state.lambda_tilde-obj.list_pi(ind_d)*(obj.alpha_p_pi+state.mu*obj.alpha_a_pi)))/(obj.alpha_p_x+obj.alpha_a_x*state.mu); 
              % apply T operator for agent
                T_value_a=log(sum(obj.PI(state.s,:).*exp(-value_a_star./obj.theta_a)));
                
                
                
                % entropy contribution for the planner
                entropy_planner=obj.beta*obj.theta_p*sum(obj.PI(state.s,:).*(obj.list_m_p_star(ind_d,:).*log(obj.list_m_p_star(ind_d,:))));
                distorted_expected_continuation_values=obj.beta*sum(obj.PI(state.s,:).*(obj.list_m_p_star(ind_d,:).*value_p_star));
                
                
                obj.list_V_a(ind_d)=-.5*(obj.alpha_a_pi*obj.list_pi(ind_d).^2+obj.alpha_a_x*(obj.list_x(ind_d)-obj.xstar).^2)-obj.beta*obj.theta_a.*T_value_a;
                
                obj.list_V_p(ind_d)=-.5*(obj.alpha_p_pi*obj.list_pi(ind_d)^2+obj.alpha_p_x*(obj.list_x(ind_d)-obj.xstar).^2)+entropy_planner+distorted_expected_continuation_values;
                
              
            obj.list_error(ind_d)=sum(abs(fvec));
            
            
        end
        
        
        
        
        
        
        
        
        
        
        function update_policy_rules(obj)
            % use the stored list of policies on the grid to update the
            % coeff.
            
            for s=1:obj.N
                
                pi_fit=obj.list_pi((s-1)*length(obj.domain)/obj.N+1:(s)*length(obj.domain)/obj.N);
                x_fit=obj.list_x((s-1)*length(obj.domain)/obj.N+1:(s)*length(obj.domain)/obj.N);
                V_a_fit=obj.list_V_a((s-1)*length(obj.domain)/obj.N+1:(s)*length(obj.domain)/obj.N);
                V_p_fit=obj.list_V_p((s-1)*length(obj.domain)/obj.N+1:(s)*length(obj.domain)/obj.N);                
                domain_fit=obj.domain((s-1)*length(obj.domain)/obj.N+1:(s)*length(obj.domain)/obj.N,1:2);                
                
                obj.coeff_pi(:,s)=funfitxy(obj.fspace{s},domain_fit,pi_fit);
                obj.coeff_V_a(:,s)=funfitxy(obj.fspace{s},domain_fit,V_a_fit);
                obj.coeff_V_p(:,s)=funfitxy(obj.fspace{s},domain_fit,V_p_fit);                
                obj.coeff_x(:,s)=funfitxy(obj.fspace{s},domain_fit,x_fit);
                
                
                obj.pi{s}=@(end_state) funeval(obj.coeff_pi(:,s), obj.fspace{s},end_state);
                obj.x{s}=@(end_state) funeval(obj.coeff_x(:,s), obj.fspace{s},end_state);
                obj.V_a{s}=@(end_state) funeval(obj.coeff_V_a(:,s), obj.fspace{s},end_state);
                obj.V_p{s}=@(end_state) funeval(obj.coeff_V_p(:,s), obj.fspace{s},end_state);
                
                
                for s_star=1:obj.N
                lambda_tilde_fit{s_star}=obj.list_lambda_tilde((s-1)*length(obj.domain)/obj.N+1:(s)*length(obj.domain)/obj.N,s_star);
                obj.coeff_lambda_tilde{s_star}(:,s)=funfitxy(obj.fspace{s},domain_fit,lambda_tilde_fit{s_star});                    
                obj.lambda_tilde{s_star,s}=@(end_state) funeval(obj.coeff_lambda_tilde{s_star}(:,s), obj.fspace{s},end_state);                                
                
                mu_fit{s_star}=obj.list_mu((s-1)*length(obj.domain)/obj.N+1:(s)*length(obj.domain)/obj.N,s_star);
                obj.coeff_mu{s_star}(:,s)=funfitxy(obj.fspace{s},domain_fit,mu_fit{s_star});                    
                obj.mu{s_star,s}=@(end_state) funeval(obj.coeff_mu{s_star}(:,s), obj.fspace{s},end_state);                                
                end
                
            end
            
            
        
        end
        
        
        function iterate_on_value_functions(obj)
            % This function will keep policy rules fixed and just iterate
            % on the value functions
            for s=1:obj.N
                domain_fit=obj.domain((s-1)*length(obj.domain)/obj.N+1:(s)*length(obj.domain)/obj.N,1:2);   
                pi_fit=obj.list_pi((s-1)*length(obj.domain)/obj.N+1:(s)*length(obj.domain)/obj.N,:);
                x_fit=obj.list_x((s-1)*length(obj.domain)/obj.N+1:(s)*length(obj.domain)/obj.N,:);
                
                for s_star=1:obj.N
                    mu_star(:,s_star)=obj.mu{s_star,s}(domain_fit);
                    lambda_tilde_star(:,s_star)=obj.lambda_tilde{s_star,s}(domain_fit);
                    pi_star(:,s_star)=obj.pi{s_star}([mu_star(:,s_star) lambda_tilde_star(:,s_star)] );                    
                    value_a_star(:,s_star)=obj.V_a{s_star}([mu_star(:,s_star) lambda_tilde_star(:,s_star)]);
                    value_p_star(:,s_star)=obj.V_p{s_star}([mu_star(:,s_star) lambda_tilde_star(:,s_star)]);                    
                    m_p_star(:,s_star)=exp( -(1/(obj.theta_p)).*(value_p_star(:,s_star)) );                                        
                end
                % apply T operator for agent
                T_value_a=log(sum(repmat(obj.PI(s,:),length(domain_fit),1).*exp(-value_a_star./obj.theta_a),2));
                
                
                
                % entropy contribution for the planner
                m_p_star=m_p_star./repmat(sum(repmat(obj.PI(s,:),length(domain_fit),1).*m_p_star,2),1,obj.N);
                entropy_planner=obj.beta*obj.theta_p*sum(repmat(obj.PI(s,:),length(domain_fit),1).*(m_p_star.*log(m_p_star)),2);
                distorted_expected_continuation_values=obj.beta*sum(repmat(obj.PI(s,:),length(domain_fit),1).*(m_p_star.*value_p_star),2);
                
                
                V_a_fit=-.5*(obj.alpha_a_pi*pi_fit.^2+obj.alpha_a_x*(x_fit-obj.xstar).^2)-obj.beta*obj.theta_a.*T_value_a;
                
                V_p_fit=-.5*(obj.alpha_p_pi*pi_fit.^2+obj.alpha_p_x*(x_fit-obj.xstar).^2)+entropy_planner+distorted_expected_continuation_values ;
                
                
                
                obj.list_V_a((s-1)*length(obj.domain)/obj.N+1:(s)*length(obj.domain)/obj.N,:)=V_a_fit;
                obj.list_V_p((s-1)*length(obj.domain)/obj.N+1:(s)*length(obj.domain)/obj.N,:)=V_p_fit;
                
            end
            
        end
        
        
        
        function plot_error_points(obj)
            
            
            for s=1:obj.N
               figure() 
                domain_fit=obj.domain((s-1)*length(obj.domain)/obj.N+1:(s)*length(obj.domain)/obj.N,1:2);                
                max_error=obj.list_error((s-1)*length(obj.domain)/obj.N+1:(s)*length(obj.domain)/obj.N);                
                error_indx=find(max_error>obj.tol);
                if ~isempty(error_indx)
                    
                error_points=domain_fit(error_indx,:);
                scatter(error_points(:,1),error_points(:,2))
                end
            end
            
            
        end
        
        
        function plot_domain=obtain_plot_domain(obj,plot_grid_size,s)
            
            
            discrete_plot_grid_lambda=linspace(obj.lambda_tilde_min(s),obj.lambda_tilde_max(s),4);
            discrete_plot_grid_mu=linspace(obj.mu_min,obj.mu_max,4);
            
            for ctr=1:4
            
            plot_domain.mu{ctr}=[linspace(obj.mu_min,obj.mu_max,plot_grid_size)' ones(plot_grid_size,1)*discrete_plot_grid_lambda(ctr)];
            
            plot_domain.lambda_tilde{ctr}=[discrete_plot_grid_mu(ctr)*ones(plot_grid_size,1) linspace(obj.lambda_tilde_min(s),obj.lambda_tilde_max(s),plot_grid_size)'];
            end
            
        end
        
        
        
        function plot_function(obj,x,name_of_function)
            plot_domain=obj.obtain_plot_domain(10,1);

            C={'b','r'}
wrt_mu=figure('Name','w.r.t mu')
wrt_lambda_tilde=figure('Name','w.r.t lambda_tilde')

if length(x)>1
    for s_star=1:length(x)

        figure(wrt_mu)
        for ctr=1:4
subplot(2,2,ctr)
x_data=plot_domain.mu{ctr}(:,1);
y_data=x{s_star}(plot_domain.mu{ctr});
plot(x_data,y_data,'linewidth',3,'color',C{s_star})
xlabel('$\mu$','interpreter','latex')
ylabel(name_of_function,'interpreter','latex')
title(strcat('$\tilde{\lambda}$= ',num2str(plot_domain.mu{ctr}(1,2))) ,'interpreter','latex' )
hold on
end    


figure(wrt_lambda_tilde)
for ctr=1:4
subplot(2,2,ctr)
x_data=plot_domain.lambda_tilde{ctr}(:,2);
y_data=x{s_star}(plot_domain.lambda_tilde{ctr});
plot(x_data,y_data,'linewidth',3,'color',C{s_star})
xlabel('$\tilde{\lambda}$','interpreter','latex')
ylabel(name_of_function,'interpreter','latex')
title(strcat('$\mu$= ',num2str(plot_domain.lambda_tilde{ctr}(1,1))) ,'interpreter','latex' )
hold on
end




    end
else
    
 
    
figure()
for ctr=1:4
subplot(2,2,ctr)
x_data=plot_domain.mu{ctr}(:,1);
y_data=x(plot_domain.mu{ctr});
plot(x_data,y_data,'linewidth',3)
xlabel('$\mu$','interpreter','latex')
ylabel(name_of_function,'interpreter','latex')
title(strcat('$\tilde{\lambda}$= ',num2str(plot_domain.mu{ctr}(1,2))) ,'interpreter','latex' )
end


figure()
for ctr=1:4
subplot(2,2,ctr)
x_data=plot_domain.lambda_tilde{ctr}(:,2);
y_data=x(plot_domain.lambda_tilde{ctr});
plot(x_data,y_data,'linewidth',3)
xlabel('$\tilde{\lambda}$','interpreter','latex')
ylabel(name_of_function,'interpreter','latex')
title(strcat('$\mu$= ',num2str(plot_domain.mu{ctr}(1,2))) ,'interpreter','latex' )
end



end


        
        
        end
    
        function [sample_path_lambda_tilde,sample_path_mu, sample_path_s]= simulate(obj,T,lambda0,mu0)
            sample_path_lambda_tilde=ones(T,1)*lambda0;
            sample_path_mu=ones(T,1)*mu0;                        
            sample_path_s=ones(T,1)*s0;
            
            for t=2:T
                sample_path_s(t)=(discretesample(obj.PI(sample_path_s(t),:),1));
                sample_path_lambda_tilde(t)=obj.lambda_tilde{sample_path_s(t-1),sample_path_s(t)}(sample_path_lambda_tilde(t-1),sample_path_mu(t-1));
                
            end
        end
        
        
        function error=error_in_pc(obj,state)
            mu_today=state.mu;
            lambda_tilde_today=state.lambda_tilde;
            s=state.s;
            
            for s_star=1:obj.N
                    value_a_star(:,s_star)=obj.V_a{s_star}([obj.mu{s_star,s}([mu_today lambda_tilde_today]) obj.lambda_tilde{s_star,s}([mu_today lambda_tilde_today])]);
                
                pi_star(s_star)=obj.pi{s_star}([obj.mu{s_star,s}([mu_today lambda_tilde_today]) obj.lambda_tilde{s_star,s}([mu_today lambda_tilde_today])]);                                
            end
            
            % m_a_star
            m_a_star=exp( -(1/(obj.theta_a)).*(value_a_star));
            m_a_star=m_a_star./sum(obj.PI(s,:).*m_a_star);                                               
            
            
            % 
            
            E_pi_star=sum(obj.PI(s,:).*m_a_star.*pi_star);
            
            other_stuff_in_rhs=(obj.kappa^2/(obj.alpha_p_x+mu_today*obj.alpha_a_x))*lambda_tilde_today+obj.kappa*obj.xstar+obj.Z(s);
            
            
            
            pi_coeff=(1+obj.kappa^2*(obj.alpha_p_pi+ mu_today*obj.alpha_a_pi)./(obj.alpha_p_x+ mu_today*obj.alpha_a_x));
            
            
            error=obj.pi{s}([mu_today lambda_tilde_today])-pi_coeff^(-1).*(other_stuff_in_rhs+obj.beta*E_pi_star);
            
            
            
            
        end
        
    end
end