close all
plot_domain=op.obtain_plot_domain(10,1)
clear x
s=1



for s_star=1:2
x{s_star}=@(end_state) (op.mu{s_star}(end_state)-end_state(:,1))
end

name_of_function='$\Delta\mu(z^*)$'
% 
  for s=1:2
  x{s}=@(end_state) (op.V_p{s}(end_state))
  end
 

% 
 name_of_function='$V_p$'
% 
 op.plot_function(x,name_of_function)
% 
% 
% 
%  for s=1:2
%  x{s}=@(end_state) (op.pi{s}(end_state))
%  end
% 
% 
% 
% name_of_function='$\pi$'
% 
% op.plot_function(x,name_of_function)
% 
